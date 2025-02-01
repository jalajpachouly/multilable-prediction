def filter_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile: # Added encoding='utf-8'
            for line in infile:
                if not line.strip().startswith("Fold"):
                    outfile.write(line)

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except UnicodeDecodeError as e:  # More specific exception handling.
        print(f"Encoding Error: Could not decode file '{input_file}' using UTF-8.  It may be encoded differently. Error details: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

#... (rest of your code remains the same)



# Example usage:
input_filename = "C:\\DB\\jalaj\\Phd-Jalaj\\various-k-values-of-feature-selection.txt"  # Replace with your input file name
output_filename = "C:\\DB\\jalaj\\Phd-Jalaj\\various-k-values-of-feature-selection-output.txt" # Replace with your desired output file name

filter_file(input_filename, output_filename)

print(f"File '{input_filename}' filtered.  Result written to '{output_filename}'")



