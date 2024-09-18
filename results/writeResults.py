import re
from collections import defaultdict

def parse_results(file_path):
    results = defaultdict(dict)
    current_model = None
    current_dataset = None
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Identify model name (e.g., CLIP, DINOV2)
            if re.match(r'^[a-zA-Z]+\w*:$', line):
                current_model = line[:-1]
            
            # Identify dataset (e.g., Downstream Evaluation on Chest X-Ray)
            elif "Downstream Evaluation on" in line:
                current_dataset = line.split(" on ")[1]
            
            # Identify metrics
            elif "{" in line and "}" in line:
                metrics = eval(line)
                if 'AUROC' in metrics:
                    results[current_model][current_dataset] = round(metrics['AUROC'], 3)
                else:
                    results[current_model][current_dataset] = round(metrics['AUROC'], 3)
    
    return results

def generate_latex_table(results):
    # Extract all datasets for columns
    datasets = ['CheXpert', 'MIMIC-CXR', 
            'BraTS-Path', 'CRC-HE',
            'DeepLesion', 'ADNI PET',
            'COVID-QU-Ex', 'ISIC', 'ADNI MRI', 'ACDC']
    
    # Start LaTeX table
    latex_table = "\\begin{tabular}{l" + "c" * len(datasets) + "}\n"
    latex_table += "Model & " + " & ".join(datasets) + " \\\\\n"
    latex_table += "\\hline\n"
    
    # Fill in data
    for model, model_data in results.items():
        row = [model]
        for dataset in datasets:
            row.append(str(model_data.get(dataset, 'N/A')))
        latex_table += " & ".join(row) + " \\\\\n"
    
    latex_table += "\\end{tabular}"
    return latex_table

def main():
    file_path = "results.txt"  # Path to your results.txt file
    results = parse_results(file_path)
    latex_table = generate_latex_table(results)
    print(latex_table)

if __name__ == "__main__":
    main()
