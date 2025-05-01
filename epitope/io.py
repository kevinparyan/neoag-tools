import os
import re
import gzip
from collections import OrderedDict
import warnings
import numpy as np
import pandas as pd

from .seq import unpad_peptide


# coding mutations
# excluding 'Splice_Site' from amino acid translation
coding_mut_class = [
    'Missense_Mutation',
    'Frame_Shift_Ins',
    'Frame_Shift_Del',
    'Nonstop_Mutation',
    'In_Frame_Ins',
    'In_Frame_Del'
]
neoorf_mut_class = [
    'Frame_Shift_Ins',
    'Frame_Shift_Del',
    'Nonstop_Mutation'
]

# maf columns
required_cols = OrderedDict({
    'Hugo_Symbol'           : str,
    'Chromosome'            : str,
    'Start_position'        : int,
    'End_position'          : int,
    'Reference_Allele'      : str,
    'Tumor_Seq_Allele'      : str,
    'Variant_Classification': str,
    'Variant_Type'          : str,
    'Genome_Change'         : str,
    'Annotation_Transcript' : str,
    'cDNA_Change'           : str,
    'Codon_Change'          : str,
    'Protein_Change'        : str,
})
optional_cols = OrderedDict({
    'ClonalStructure': np.dtype('object'),
    'PhaseID'        : float,
    'GermlineID'     : float
})

# peptide and neoorf file columns
peptide_header = ['Hugo_Symbol',
                  'pep_wt',
                  'pep',
                  'ctex_up',
                  'ctex_dn',
                  'pep_start',
                  'pep_end']

neoorf_header = ['Hugo_Symbol',
                 'upstream',
                 'neoORF',
                 'neoORF_start']

common_header = ['transcript_id',
                 'gene_id',
                 'TPM',
                 'Tumor_Sample_Barcode',
                 'clone',
                 'Chromosome',
                 'Start_position',
                 'End_position',
                 'Variant_Classification',
                 'Variant_Type',
                 'Genome_Change',
                 'cDNA_Change',
                 'Codon_Change',
                 'Protein_Change']


def read_maf(maf, name, name_col):
    if maf == None:
        return pd.DataFrame()
    print(f"reading MAF: {maf}")
    # reading MAF in a primivie way to handle commend characters '#'
    # in the middle of the line which pd.read_csv couldn't handle
    # if Start_Position or End_Position are __UNKNOWN__ skip the line
    print("trying to implement fix for empty Start_position or end_position")
    header = []
    record = []
    for line in open(maf):
        if line.startswith('#'):
            continue
        line = line.rstrip('\n').split('\t')
        if line[0] == 'Hugo_Symbol':
            header = line
            start_index = None
            end_index = None
            for index, item in enumerate(line):
                if item == 'Start_Position' or item == 'Start_position':
                    start_index = index
                    break
            for index, item in enumerate(line):
                if item == 'End_Position' or item == 'End_position':
                    end_index = index
                    break
        if line[start_index] == '__UNKNOWN__' or line[end_index] == '__UNKNOWN__':
            continue
        #elif not line[start_index]:
        #    print(f"start position empty: {line[start_index]}")
        #    continue
        #elif not line[end_index]:
        #    print(f"end position empty: {line[end_index]}")
        #    continue
        else:
            record.append({h:x for h,x in zip(header,line)})
    df = pd.DataFrame(data=record, dtype=str)

    tid = set(df[name_col])
    if name not in tid:
        raise Exception(name + " not in " + os.path.basename(maf))
    if len(tid) > 1:
        row = df[name_col] == name
        df[row].reset_index(drop=True, inplace=True)
    tid = name

    # rename MAF header
    df.rename(columns={'Tumor_Seq_Allele2':'Tumor_Seq_Allele'}, inplace=True)
    if 'Start_Position' in df:
        df.rename(columns={'Start_Position':'Start_position'}, inplace=True)
    if 'End_Position' in df:
        df.rename(columns={'End_Position':'End_position'}, inplace=True)
    # Convert Start_position and End_position to numeric, handling errors
    for col in ['Start_position', 'End_position']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(subset=[col], inplace=True)  # Drop rows with NaN in critical position columns

    # format MAF columns
    if 'PhaseID' in df:
        df.loc[df['PhaseID'].isin(['nan','']), 'PhaseID'] = np.nan
    if 'GermlineID' in df:
        df.loc[df['GermlineID'].isin(['nan','']), 'GermlineID'] = np.nan
    
    # subset MAF columns
    maf_cols = required_cols.copy()
    for oc in optional_cols:
        if oc in df:
            maf_cols[oc] = optional_cols[oc]
    try:
        # uncomment for troubleshooting purposes
        #ls = [type(item) for item in maf_cols]
        #print(f"maf_cols.keys(): {maf_cols.keys()}")
        #print(f"maf_cols: {maf_cols}")
        #print(f"df cols dtypes: {df.dtypes}")
        #print(f"maf_cols types: {ls}")
    # Iterate through columns individually to allow us to catch specific issues if they arise
        for col in maf_cols.keys():
            print(f"Converting column: {col} to type: {maf_cols[col]}")
            df[col] = df[col].astype(maf_cols[col])

    except Exception as e:
        print(df[col])
        # Print which column caused the error and provide a descriptive message
        problematic_col = col  # Capture the last column being processed
        raise KeyError(f"Error converting column '{problematic_col}' - {str(e).replace('index', 'MAF')}")

    # process clone information if supplied
    if 'ClonalStructure' in df:
        df.loc[df['ClonalStructure'].isin(['nan','']), 'ClonalStructure'] = np.nan
        i = ~df['ClonalStructure'].isna()
        make_set = lambda x: set([xi.strip() for xi in re.sub('\[|\]','',x).split(',') if xi.strip().isnumeric()])
        df.loc[i, 'ClonalStructure'] = df.loc[i, 'ClonalStructure'].apply(make_set)
        
    return df


def read_rsem_gene(rsem_gene, transcript_list=[]):
    """
    Returns Transcript ID (without versioning) and TPM
    from RSEM gene expression matrix
    """
    # strip off transcript ID versions
    if transcript_list != []:
        transcript_list = set([txid.split('.')[0] for txid in transcript_list])

    gene_tpm = dict()
    try:
        if rsem_gene.endswith('.gz'):
            rsem_gene_fi = gzip.open(rsem_gene, 'rt')
        else:
            rsem_gene_fi = open(rsem_gene, 'r')
    except IOError:
        raise Exception('cannot read: {}'.format(rsem_gene))

    header = []
    for row in rsem_gene_fi:
        # skip comment lines if any
        if row.startswith('#'):
            continue
        row = row.rstrip('\n').split('\t')
        if row[0] == 'gene_id':
            continue
        
        transcript_ids = set([txid.split('.')[0] for txid in row[1].split(',')])
        tpm = row[5]
        
        if transcript_list != []:
            overlap = transcript_list.intersection(transcript_ids)
            if len(overlap) == 0:
                continue
            transcript_ids = overlap

        for txid in transcript_ids:
            gene_tpm.setdefault(txid, tpm)

    return gene_tpm


def get_fasta_header(m, name, txid, clone=None):
    gene = m['Hugo_Symbol'].unique()[0]
    nt = ';'.join(m['cDNA_Change'].unique())
    aa = ';'.join(m['Protein_Change'].unique())
    if clone != None:
        return '|'.join([name, txid, gene, nt, aa, 'clone='+clone])
    else:
        return '|'.join([name, txid, gene, nt, aa])


def write_fasta(fo, tumor_name, normal_name, muts, gmuts,
                txid, wt, mt, aa_str, mt_str, exon_str, clone):
    """
    Write peptide and cDNA fasta.

    Each line per record represents:
      1. FASTA entry header
      2. wild type protein sequence
      3. amino acid concordance string
      4. mutated protein sequence
      5. mutated cDNA sequence
      6. cDNA concordance string
      7. wild type cDNA sequence
      8. exon boundary string

    FASTA entry header is formatted:
      > {somatic_mutation_field} ; {germline_mutation_field}
 
    Germline mutation filed will be only written if there are
    phased coding germline mutations.
    """
    tumor_field = get_fasta_header(muts, tumor_name, txid, clone)
    normal_field = ''
    if not gmuts.empty:
        normal_field = get_fasta_header(gmuts, normal_name, txid)
    header = '>' + tumor_field + ';' + normal_field
    fo.write(header+'\n')
    fo.write(wt.aa+'\n')
    fo.write(aa_str+'\n')
    fo.write(mt.aa+'\n')
    fo.write(mt.seq+'\n')
    fo.write(mt_str+'\n')
    fo.write(wt.seq+'\n')
    fo.write(exon_str+'\n')


#def write_peptide(fo, smuts, clone, wt, mt, mstr, idx,
#                  name, txid, gnid, tpm, flen, pep_lens):
#    flank_length = flen+max(pep_lens)-1
#    # --- Early check for empty mstr ---
#    if not mstr:
#        warnings.warn(f"Skipping peptides for {name}/{clone} - mstr is empty.")
#        return # Exit the function if mstr is empty
#    mstr_len = len(mstr)
#    for index, m in smuts.iterrows():
#        try: # Add a try block for unexpected issues within the loop iteration
#
#		start, end = idx[index]
#		start = start - start%3 + 1
#		end = (end-1) - (end-1)%3 + 1
#	 # --- Check original coordinates from idx ---
#                if start >= mstr_len or end > mstr_len or start < 0 or end <= 0:
#                    warnings.warn(f"Skipping peptide for mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: "
#                               f"Initial coordinates ({orig_start}, {orig_end}) out of bounds for mstr length {mstr_len}.")
#                continue # Skip to the next mutation in the loop	
#		# adjust reading frame for shifted amino acid mismatches
#		for i in range(start,len(mstr),3):
#                    if i >= mstr_len: break
#		    if mstr[i] == '*':
#			start = i
#			break
#		for i in range(end,start-3,-3):
#                    # Check 'i' validity
#                    if i >= mstr_len or i < 0 : continue # Skip invalid index within loop
#		    if mstr[i] == '*':
#			end = i
#		# --- Final check on start/end after potential modification by loops ---
#                if start >= mstr_len or end >= mstr_len or start < 0 or end < 0 or end < start : # Check if end is before start
#                    warnings.warn(f"Skipping peptide for mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: "
#                               f"Final coordinates after stop search ({start}, {end}) are invalid or out of order for mstr length {mstr_len}.")
#                    continue # Skip to the next mutation
#
#		if m['Variant_Classification'].startswith('Frame_Shift'):
#		    wt_aa = unpad_peptide(wt.aa[start:])
#		    mt_aa = unpad_peptide(mt.aa[start:])
#		    # make wild type sequence same length as mutated sequence
#		    wt_aa += '-'*(len(mt_aa)-len(wt_aa))
#		    wt_aa = wt_aa[:len(mt_aa)]
#		    mt_aa_dn = ''
#		    wt_aa_dn = ''
#		elif m['Variant_Classification'].startswith('Nonstop_Mutation'):
#		    mt_aa = unpad_peptide(mt.aa[start:])
#		    wt_aa = '-'*len(mt_aa)
#		    mt_aa_dn = ''
#		    wt_aa_dn = ''            
#		else:
#		    wt_aa = unpad_peptide(wt.aa[start:end+1])
#		    mt_aa = unpad_peptide(mt.aa[start:end+1])
#		    mt_aa_dn = unpad_peptide(mt.aa[(end+3):])
#		    wt_aa_dn = unpad_peptide(wt.aa[(end+3):])
#		    
#		    # trim downstream sequence with specified flanking peptide length
#		    mt_aa_dn = mt_aa_dn[:flank_length] + '-'*(flank_length-len(mt_aa_dn))
#		    wt_aa_dn = wt_aa_dn[:flank_length] + '-'*(flank_length-len(wt_aa_dn))
#
#		mt_aa_up = unpad_peptide(mt.aa[:start])
#		wt_aa_up = unpad_peptide(wt.aa[:start])
#
#		# mutated protein sequence coordinates
#		aa_start = len(mt_aa_up) + 1
#		aa_stop = aa_start + len(mt_aa) - 1
#
#		# trim upstream sequence with specified flanking peptide length
#		mt_aa_up = '-'*(flank_length-len(mt_aa_up)) + mt_aa_up[-flank_length:]
#		wt_aa_up = '-'*(flank_length-len(wt_aa_up)) + wt_aa_up[-flank_length:]
#		
#		start = len(mt_aa_up)
#		stop = start + len(mt_aa)
#		
#		wt_aa = wt_aa_up + wt_aa + wt_aa_dn
#		mt_aa = mt_aa_up + mt_aa + mt_aa_dn
#		for plen in pep_lens:
#		    for i in range(start-plen+1, stop):
#			pep_wt = wt_aa[i:i+plen]
#			pep = mt_aa[i:i+plen]
#			if len(pep) < plen:
#			    continue
#			elif '-' in pep:
#			    continue
#			elif pep == pep_wt:
#			    continue
#			ctex_up = mt_aa[i-flen:i]
#			ctex_dn = mt_aa[i+plen:i+plen+flen]
#			ctex_dn += '-'*(flen-len(ctex_dn))
#			fo.write('\t'.join([
#			    m['Hugo_Symbol'],
#			    pep_wt, pep, ctex_up, ctex_dn,
#			    str(aa_start),
#			    str(aa_stop),
#			    txid, gnid, tpm, name, clone,
#			    m['Chromosome'],
#			    str(m['Start_position']),
#			    str(m['End_position']),
#			    m['Variant_Classification'],
#			    m['Variant_Type'],
#			    m['Genome_Change'],
#			    m['cDNA_Change'],
#			    m['Codon_Change'],
#			    m['Protein_Change'],
#			])+'\n')
#    except IndexError as e:
#        warnings.warn(f"Caught IndexError processing mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: {e}. Skipping.")
#        continue # Skip to next mutation on unexpected index error
#    except KeyError as e:
#        warnings.warn(f"Caught KeyError processing mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: Missing key {e}. Skipping.")
#        continue # Skip to next mutation if a key is missing from 'm' or 'idx'
#    except Exception as e:
#        warnings.warn(f"Caught unexpected Exception processing mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: {type(e).__name__} - {e}. Skipping.")
#        continue # Skip on any other error for this mutation

def write_peptide(fo, smuts, clone, wt, mt, mstr, idx,
                  name, txid, gnid, tpm, flen, pep_lens):
    flank_length = flen + max(pep_lens) - 1

    # --- Early check for empty mstr ---
    if not mstr:
        warnings.warn(f"Skipping peptides for {name}/{clone} - mstr is empty.")
        return # Exit the function if mstr is empty

    mstr_len = len(mstr) # Store length for efficiency

    for index, m in smuts.iterrows():
        try: # Add a try block for unexpected issues within the loop iteration
            orig_start, orig_end = idx[index] # Get original values first

            # --- Check original coordinates from idx ---
            if orig_start >= mstr_len or orig_end > mstr_len or orig_start < 0 or orig_end <= 0:
                 warnings.warn(f"Skipping peptide for mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone} transcript ID {txid} gene ID {gnid}: "
                               f"Initial coordinates ({orig_start}, {orig_end}) out of bounds for mstr length {mstr_len}.")
                 continue # Skip to the next mutation in the loop

            start = orig_start - orig_start % 3 + 1
            end = (orig_end - 1) - (orig_end - 1) % 3 + 1

            # --- Check coordinates AFTER modulo adjustment ---
            # Ensure start and end point within or at the boundaries validly
            if start >= mstr_len or end >= mstr_len or start < 0 or end < 0:
                warnings.warn(f"Skipping peptide for mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone} transcript ID {txid} gene ID {gnid}: ")
                warnings.warn(f"Adjusted coordinates ({start}, {end}) out of bounds for mstr length {mstr_len}.")
                continue # Skip to the next mutation

            # --- Clamp end to be within bounds if necessary for range start ---
            # The range function itself handles the upper bound, but ensure the start of the range is valid.
            safe_end_for_loop2 = min(end, mstr_len - 1)

            # --- Find first stop codon after start ---
            # Make sure the loop doesn't start out of bounds
            first_stop_found = False
            if start < mstr_len:
                 for i in range(start, mstr_len, 3):
                     # Check 'i' itself just in case, although range should handle it
                     if i >= mstr_len: break
                     if mstr[i] == '*':
                         start = i # Update start to the stop codon position
                         first_stop_found = True
                         break

            # --- Find last stop codon before the (potentially updated) start ---
            # Make sure the loop doesn't start out of bounds
            last_stop_found = False
            # Ensure start index for range() is not negative
            if safe_end_for_loop2 >= 0:
                # Iterate down to index 0. The stop parameter is exclusive.
                for i in range(safe_end_for_loop2, start - 3, -3):
                    # Check 'i' validity
                    if i >= mstr_len or i < 0 : continue # Skip invalid index within loop
                    if mstr[i] == '*':
                        end = i # Update end to the stop codon position
                        last_stop_found = True
                        # NOTE: This loop finds the LAST stop codon before start.
                        # Depending on logic, maybe you only wanted the FIRST one counting down?
                        # If so, add 'break' here. Assuming current logic is correct.

            # --- Final check on start/end after potential modification by loops ---
            if start >= mstr_len or end >= mstr_len or start < 0 or end < 0 or end < start : # Check if end is before start
                 warnings.warn(f"Skipping peptide for mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone} transcript ID {txid} gene ID {gnid}: ")
                 warnings.warn(f"Final coordinates after stop search ({start}, {end}) are invalid or out of order for mstr length {mstr_len}.")
                 continue # Skip to the next mutation


            # --- REST OF YOUR ORIGINAL CODE FOR PROCESSING ---
            # (Variant Classification checks, slicing, writing)

            if m['Variant_Classification'].startswith('Frame_Shift'):
                # Ensure slices use the potentially updated start/end
                wt_aa = unpad_peptide(wt.aa[start:])
                mt_aa = unpad_peptide(mt.aa[start:])
                # make wild type sequence same length as mutated sequence
                wt_aa += '-'*(len(mt_aa)-len(wt_aa))
                wt_aa = wt_aa[:len(mt_aa)]
                mt_aa_dn = ''
                wt_aa_dn = ''
            elif m['Variant_Classification'].startswith('Nonstop_Mutation'):
                mt_aa = unpad_peptide(mt.aa[start:])
                wt_aa = '-'*len(mt_aa)
                mt_aa_dn = ''
                wt_aa_dn = ''
            else:
                # Check slice boundaries carefully based on updated start/end
                slice_end = min(end + 1, len(wt.aa)) # Ensure slice end is valid
                slice_end_plus_3 = min(end + 3, len(wt.aa))

                wt_aa = unpad_peptide(wt.aa[start:slice_end])
                mt_aa = unpad_peptide(mt.aa[start:slice_end])
                mt_aa_dn = unpad_peptide(mt.aa[slice_end_plus_3:]) # Use end+3 adjusted
                wt_aa_dn = unpad_peptide(wt.aa[slice_end_plus_3:]) # Use end+3 adjusted

                # trim downstream sequence with specified flanking peptide length
                mt_aa_dn = mt_aa_dn[:flank_length] + '-'*(flank_length-len(mt_aa_dn))
                wt_aa_dn = wt_aa_dn[:flank_length] + '-'*(flank_length-len(wt_aa_dn))

            # Ensure slice boundary is valid
            slice_start = min(start, len(mt.aa))
            mt_aa_up = unpad_peptide(mt.aa[:slice_start])
            wt_aa_up = unpad_peptide(wt.aa[:slice_start])


            # mutated protein sequence coordinates
            aa_start = len(mt_aa_up) + 1
            aa_stop = aa_start + len(mt_aa) - 1

            # trim upstream sequence with specified flanking peptide length
            mt_aa_up = '-'*(flank_length-len(mt_aa_up)) + mt_aa_up[-flank_length:]
            wt_aa_up = '-'*(flank_length-len(wt_aa_up)) + wt_aa_up[-flank_length:]

            start_offset = len(mt_aa_up) # Renaming 'start' used for slicing loops
            stop_offset = start_offset + len(mt_aa) # Renaming 'stop' used for slicing loops

            wt_aa_full = wt_aa_up + wt_aa + wt_aa_dn # Renamed variable
            mt_aa_full = mt_aa_up + mt_aa + mt_aa_dn # Renamed variable

            for plen in pep_lens:
                # Adjust loop range based on renamed variables
                for i in range(start_offset - plen + 1, stop_offset):
                    if i < 0: continue # Prevent negative index slicing
                    pep_wt = wt_aa_full[i : i + plen]
                    pep = mt_aa_full[i : i + plen]
                    if len(pep) < plen:
                        continue
                    elif '-' in pep:
                        continue
                    # Small optimization: check identity only if lengths match expected plen
                    elif len(pep_wt) == plen and pep == pep_wt:
                         continue

                    # Context slicing bounds checks
                    ctex_up_start = max(0, i - flen)
                    ctex_dn_end = min(len(mt_aa_full), i + plen + flen)

                    ctex_up = mt_aa_full[ctex_up_start : i]
                    ctex_dn = mt_aa_full[i + plen : ctex_dn_end]
                    ctex_dn += '-'*(flen-len(ctex_dn))

                    # Ensure all required fields exist in 'm' before writing
                    required_fields = ['Hugo_Symbol', 'Chromosome', 'Start_position', 'End_position',
                                     'Variant_Classification', 'Variant_Type', 'Genome_Change',
                                     'cDNA_Change', 'Codon_Change', 'Protein_Change']
                    if not all(field in m for field in required_fields):
                        warnings.warn(f"Skipping write for mutation index {index} due to missing fields.")
                        continue # Skip writing if data is incomplete

                    fo.write('\t'.join([
                        str(m['Hugo_Symbol']), # Ensure strings
                        str(pep_wt), str(pep), str(ctex_up), str(ctex_dn),
                        str(aa_start), str(aa_stop),
                        str(txid), str(gnid), str(tpm), str(name), str(clone),
                        str(m['Chromosome']),
                        str(m['Start_position']), str(m['End_position']),
                        str(m['Variant_Classification']), str(m['Variant_Type']),
                        str(m['Genome_Change']), str(m['cDNA_Change']),
                        str(m['Codon_Change']), str(m['Protein_Change']),
                    ])+'\n')

        except IndexError as e:
             warnings.warn(f"Caught IndexError processing mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: {e}. Skipping.")
             continue # Skip to next mutation on unexpected index error
        except KeyError as e:
             warnings.warn(f"Caught KeyError processing mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: Missing key {e}. Skipping.")
             continue # Skip to next mutation if a key is missing from 'm' or 'idx'
        except Exception as e:
             warnings.warn(f"Caught unexpected Exception processing mutation index {index} ({m.get('Protein_Change', 'N/A')}) in {name}/{clone}: {type(e).__name__} - {e}. Skipping.")
             continue # Skip on any other error for this mutation

def write_neoorf(fo, smuts, clone, wt, mt, mstr, idx, name, txid, gnid, tpm):
    for index, m in smuts.iterrows():
        start, end = idx[index]
        start = start - start%3 + 1
        end = (end-1) - (end-1)%3 + 1

        if m['Variant_Classification'].startswith('Frame_Shift'):
            # adjust reading frame for shifted amino acid mismatches
            for i in range(start,len(mstr),3):
                if mstr[i] == '*':
                    start = i
                    break
        neoorf = unpad_peptide(mt.aa[start:])
        upstream = unpad_peptide(mt.aa[:start])
        neoorf_start = len(upstream) + 1

        fo.write('\t'.join([
            m['Hugo_Symbol'],
            upstream, neoorf, str(neoorf_start),
            txid, gnid, tpm, name, clone,
            m['Chromosome'],
            str(m['Start_position']),
            str(m['End_position']),
            m['Variant_Classification'],
            m['Variant_Type'],
            m['Genome_Change'],
            m['cDNA_Change'],
            m['Codon_Change'],
            m['Protein_Change']
        ])+'\n')
