def extract_terms(file_path):
    """
    Extracts terms from the 'FINAL ENUMERATED TERMS' section of a file.
    Returns a list of terms as strings.
    """
    terms = []
    with open(file_path, "r") as f:
        lines = f.readlines()

    capture = False
    for line in lines:
        if line.strip().startswith("FINAL ENUMERATED TERMS:"):
            capture = True
            continue
        if capture:
            if line.strip() == "" or line.strip().startswith("FINAL FORMULA"):
                break
            # Extract the part after "Term N:"
            term = line.split(":", 1)[1].strip()
            terms.append(term)

    return terms


# Example usage:
#file1_terms = extract_terms("output1.txt")
#file2_terms = extract_terms("output2.txt")

