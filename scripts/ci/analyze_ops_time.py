#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import re
import csv
import glob
import logging
from pathlib import Path
from collections import defaultdict


script_dir = os.path.dirname(os.path.abspath(__file__))

build_dir = os.path.join(script_dir, "../../build/")
build_dir = os.path.abspath(build_dir)
workdir = os.path.join(build_dir, "binary")
workdir = os.path.abspath(workdir)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_txt_content(content):
    """Parse txt file content and extract required fields"""
    pattern = (
        r'Build \[([^\]]+)\] op_name \[([^\]]+)\] index \[([^\]]+)\] '
        r'bin_file \[([^\]]+)\] duration \[([^\]]+)\] size \[([^\]]+)\] '
        r'start_time \[([^\]]+)\] end_time \[([^\]]+)\]'
    )
    match = re.search(pattern, content)
    
    if not match:
        return None
    
    return {
        'soc': match.group(1),
        'op_name': match.group(2),
        'index': match.group(3),
        'bin_file': match.group(4),
        'duration': match.group(5),
        'size': match.group(6),
        'start_time': match.group(7),
        'end_time': match.group(8)
    }


def remove_apt_suffix(op_name):
    """If op_name end with '_apt' remove it"""
    if op_name.endswith('_apt'):
        return op_name[:-4]
    return op_name


def extract_capital_name_from_binfile(bin_file):
    """Extract capital_name from bin_file format {capital_name}_hashkey"""
    # Match pattern: any characters before the first underscore
    match = re.match(r'([^_]+)_', bin_file)
    if match:
        return match.group(1)
    else:
        return None


def count_sh_files(base_dir, capital_name, op_name):
    """Count total number of {capital_name}-{op_name}-{index}.sh files"""
    # Build matching pattern
    pattern = os.path.join(base_dir, f"{capital_name}-{op_name}-*.sh")
    
    # Use glob to find matching files
    matching_files = glob.glob(pattern)
    
    return len(matching_files)


def find_txt_files(filter_names: list = None):
    """Find all txt files in specified directory structure"""
    txt_files = []
    
    # Get build/binary directory under current directory
    base_dir = workdir
    
    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist")
        return []
    
    # Iterate through all soc directories
    for soc_dir in os.listdir(base_dir):
        gen_dir = os.path.join(base_dir, soc_dir, 'gen')
        # Find all txt files in gen directory
        for file in os.listdir(gen_dir):
            match_txt = re.match(r'([^-]+)-(\d+)\.txt', file)
            if match_txt:
                op_name = match_txt.group(1)
                if op_name not in filter_names or filter_names is None:
                    txt_files.append(os.path.join(gen_dir, file))
    
    return txt_files


def process_txt_file(txt_files, merge_names):
    data_list = []
    merge_dict = defaultdict(lambda: {'duration': 0})
    for txt_file in txt_files:
        try:
            # Extract op_name and index from filename
            filename = os.path.basename(txt_file)
            # Use regex to match {op_name}-{index}.txt format
            filename_match = re.match(r'([^-]+)-(\d+)\.txt', filename)
            if not filename_match:
                logger.warning(f"Filename format mismatch: {filename}")
                continue

            # Read file content
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Parse content
            parsed_data = parse_txt_content(content)
            if not parsed_data:
                logger.warning(f"File content format mismatch: {txt_file}")
                continue

            # Extract capital name from bin_file
            bin_file = parsed_data['bin_file']
            capital_name = extract_capital_name_from_binfile(bin_file)
            if not capital_name:
                logger.warning(f"Cannot extract capital_name from bin_file {bin_file}")
                continue
            
            # Get base directory for .sh files (same as txt file directory)
            base_dir = os.path.dirname(txt_file)
            
            # Count total number of {capital_name}-{op_name}-{index}.sh files
            ops_kernel_num = count_sh_files(base_dir, capital_name, parsed_data['op_name'])
            new_kernel_name = f"{remove_apt_suffix(parsed_data['op_name'])},{ops_kernel_num}-{parsed_data['index']}"   
            
            # Build data row
            if parsed_data['op_name'] in merge_names:
                key = (parsed_data['soc'], parsed_data['op_name'])
                merge_dict[key]['duration'] += float(parsed_data['duration'])
                merge_dict[key]['size'] = parsed_data['size']
                merge_dict[key]['soc'] = parsed_data['soc']
            else:
                data_row = {
                    'op_name': new_kernel_name,
                    'duration': int(float(parsed_data['duration'])),
                    'size': parsed_data['size'],
                    'soc': parsed_data['soc'],
                }
                data_list.append(data_row)     
        except Exception as e:
            logger.error(f"Error processing file {txt_file}: {e}")
            continue
    return data_list, merge_dict


def main():
    filter_names = ["moe_gather_v2", "moe_inplace_index_add", "moe_inplace_index_add_with_sorted", "moe_masked_scatter"]
    merge_names = ["moe_token_permute_with_routing_map", "moe_token_permute_with_routing_map_grad", 
                   "moe_token_unpermute_with_routing_map"]
    # Get all txt files in current directory
    txt_files = find_txt_files(filter_names)
    
    if not txt_files:
        logger.info("No txt files found")
        return
    
    data_list, merge_dict = process_txt_file(txt_files, merge_names)

    # Append merged rows
    for (soc, op_name), values in merge_dict.items():
        merged_row = {
            'op_name': f"{remove_apt_suffix(op_name)},1-0",
            'duration': int(values['duration']),
            'size': values['size'],
            'soc': soc
        }
        data_list.append(merged_row)
    
    # Sort by op_name
    data_list.sort(key=lambda x: x['op_name'])
    
    # Write to CSV file
    if data_list:
        soc_version = data_list[0]['soc']
        csv_filename = os.path.join(build_dir, f'ops_cost_{soc_version}.csv')
        fieldnames = ['op_name', 'duration', 'size', 'soc']
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
        
        logger.info(f"Successfully processed {len(data_list)} files, results saved to {csv_filename}")
        
        # Print statistics
        logger.info(f"Statistics:")
        logger.info(f"Total records: {len(data_list)}")
    else:
        logger.info("No valid data files found")


if __name__ == "__main__":
    main()