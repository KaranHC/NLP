import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def formatting_func(ocr_text: str) -> str:
    """
    Formats the input OCR text into a prompt for the language model.

    Args:
    ocr_text (str): The OCR text from which claims are to be extracted.

    Returns:
    str: The formatted prompt string.
    """
    prompt = f"""# CONTEXT
The objective is to extract claims written in English from an Optical Character Recognition (OCR) of a product, specifically identifying the claims that the product makes.

# OBJECTIVE
You are an expert at extracting claims from a given product OCR. The primary objective is to provide a list of extracted English language claims based on the given product OCR.

# STYLE
Ensure the listed claims follow the specified format: -claim 1, -claim 2,..., -claim n. Provide precise English claims without elaboration. If no claims are present, return "-None." Avoid additional explanations or details.

# STRICT INSTRUCTION
Give only claims present in the input, do not elaborate/complete it.
Exclude Directions, Utilisation, Volume, Quantity, Expiry, Did you know, or Directions to use from consideration as claims.
Do not include nutritional information or contact details.
The texts under: Usage, how to reuse, how to use, Utilisation, Ingredients, Directions to use, Key Ingredients, Cautions, Disclaimer, Warning should not be considered as claims.

# FOCUS ON CLAIM CONTENT
Extract claims without altering or reframing the content. Only include text that directly represents a claim, avoiding elaborations or interpretations. Focus solely on the extraction of claims and refrain from rephrasing them.

# RESPONSE
Ensure the listed claims follow the specified format: -claim 1, -claim 2,..., -claim n. If no claims are present, return "-None." Avoid any additional explanations or elaborations.

# DISCLAIMER
If there are no claims, please return "-None" and nothing else, and if there are no English claims, return "-None."
PLEASE DO NOT GENERATE ANY PYTHON JUST EXTRACT THE CLAIMS

# PRODUCT OCR
{ocr_text}

# Extracted Claims ARE
"""
    return prompt

def get_response(prompt: str) -> str:
    """
    Generates a response based on the provided prompt using a pre-trained language model.

    Args:
    prompt (str): The formatted prompt string.

    Returns:
    str: The generated response containing the extracted claims.
    """
    set_seed(2024)

    model_checkpoint = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                                 trust_remote_code=True,
                                                 torch_dtype="auto",
                                                 device_map="cuda")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, do_sample=True, max_new_tokens=120)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main(ocr_text_path: str):
    """
    Main function to extract claims from a product OCR text file.

    Args:
    ocr_text_path (str): The file path of the OCR text file.
    """
    try:
        # Read OCR text from file
        with open(ocr_text_path, 'r') as file:
            ocr_text = file.read()

        # Generate prompt and extract claims
        prompt = formatting_func(ocr_text)
        response = get_response(prompt)
        
        # Log the extracted claims
        logging.info(f"Extracted Claims: {response}")

    except Exception as e:
        logging.error(f"Error processing OCR text: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_claims.py <path_to_ocr_text>")
    else:
        ocr_text_path = sys.argv[1]
        main(ocr_text_path)

# !pip install -q -U bitsandbytes
# !pip install -q -U git+https://github.com/huggingface/transformers.git
# !pip install -q -U git+https://github.com/huggingface/peft.git
# !pip install -q -U git+https://github.com/huggingface/accelerate.git
# !pip install -q -U datasets scipy ipywidgets matplotlib einops