input_file = 'dataset.txt'
output_file = 'dataset_new.txt'
prefix = 'images/'

# Read, modify, and write the file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Strip existing whitespace/newlines and add prefix
        clean_line = line.strip()
        if clean_line:  # Ensure the line is not empty
            outfile.write(f"{prefix}{clean_line}\n")

print(f"Successfully processed files to {output_file}")