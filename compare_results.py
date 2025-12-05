from pathlib import Path
import json
import re

# Parse the model output file or run results
def parse_model_output(output_file):
    """Parse model output from a text file"""
    results = {}
    with open(output_file, 'r') as f:
        content = f.read()
        # Parse format: "filename: count" or "filename: {json...}"
        lines = content.strip().split('\n')
        current_file = None
        for line in lines:
            if ':' in line and any(ext in line for ext in ['.png', '.jpg', '.jpeg']):
                parts = line.split(':', 1)
                filename = parts[0].strip()
                result = parts[1].strip()
                
                # Try to extract just the number
                if result.isdigit():
                    results[filename] = int(result)
                else:
                    # Try to extract total_pins from JSON
                    try:
                        json_match = re.search(r'\{.*\}', result, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                            results[filename] = data.get('total_pins')
                        else:
                            results[filename] = result
                    except:
                        results[filename] = result
    return results

def compare_results(ground_truth, model_results):
    """Compare ground truth with model predictions"""
    print("\n" + "="*80)
    print("PIN COUNT COMPARISON")
    print("="*80)
    print(f"{'Image Name':<50} {'Ground Truth':>12} {'Model':>12} {'Status':>10}")
    print("-"*80)
    
    correct = 0
    incorrect = 0
    uncertain = 0
    
    for filename, true_count in ground_truth.items():
        model_count = model_results.get(filename, "MISSING")
        
        if true_count is None:
            status = "NO GT"
            print(f"{filename:<50} {'NOT SET':>12} {str(model_count):>12} {status:>10}")
            uncertain += 1
        elif model_count == "MISSING":
            status = "NO OUTPUT"
            print(f"{filename:<50} {str(true_count):>12} {'MISSING':>12} {status:>10}")
            incorrect += 1
        elif str(model_count) == str(true_count):
            status = "✓ CORRECT"
            print(f"{filename:<50} {str(true_count):>12} {str(model_count):>12} {status:>10}")
            correct += 1
        else:
            status = "✗ WRONG"
            print(f"{filename:<50} {str(true_count):>12} {str(model_count):>12} {status:>10}")
            incorrect += 1
    
    print("-"*80)
    total = correct + incorrect
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\nResults: {correct}/{total} correct ({accuracy:.1f}% accuracy)")
        print(f"Incorrect: {incorrect}, Uncertain/No GT: {uncertain}")
    else:
        print("\nNo comparisons available - please set ground truth values")
    print("="*80)

if __name__ == "__main__":
    from ground_truth import GROUND_TRUTH
    
    print("To use this comparison tool:")
    print("1. First, manually count pins in each image and update ground_truth.py")
    print("2. Run your model and save output to model_output.txt")
    print("3. Run this script: python compare_results.py")
    print("\nOr provide model results as a dictionary:")
    
    # Example usage with manual input
    model_results = {
        # Add your model results here, or load from file
        # "ACS712T.png": 16,
        # "CoolMOS™ 8.png": 3,
    }
    
    if model_results:
        compare_results(GROUND_TRUTH, model_results)
    else:
        print("\nNo model results provided yet.")
