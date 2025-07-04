import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TargetedCodeFixer:
    def __init__(self, model_id="deepseek-ai/deepseek-coder-1.3B-instruct", cleanup_fixed_files=True):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.cleanup_fixed_files = cleanup_fixed_files
        self.setup_model()
    
    def setup_model(self):
        """Initialize tokenizer and model with fallback to CPU if GPU fails."""
        logging.info(f"Loading {self.model_id} on {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map=self.device.type,
                trust_remote_code=True
            )
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model on {self.device}: {e}")
            if self.device.type != "cpu":
                logging.info("Falling back to CPU...")
                try:
                    self.device = torch.device("cpu")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True
                    )
                    logging.info("Model loaded on CPU")
                except Exception as e2:
                    logging.error(f"Failed to load model on CPU: {e2}")
                    raise
            else:
                raise
    
    def extract_problematic_section(self, full_code, error_msg):
        """Extract the problematic section based on error message."""
        lines = full_code.splitlines()
        
        # Look for line numbers in error message
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            error_line = int(line_match.group(1)) - 1
            start_line = max(0, error_line - 15)
            end_line = min(len(lines), error_line + 15)
            return '\n'.join(lines[start_line:end_line]), start_line, end_line
        
        # Look for function/class names
        for i, line in enumerate(lines):
            if 'def ' in line or 'class ' in line:
                indent_level = len(line) - len(line.lstrip())
                context_end = i + 1
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= indent_level:
                        if not lines[j].strip().startswith(('"""', "'''")):
                            context_end = j
                            break
                    context_end = j + 1
                context = '\n'.join(lines[i:context_end])
                if any(keyword in error_msg.lower() for keyword in ['vmobject', 'mobject', 'vgroup', 'text', 'rectangle', 'scene']):
                    return context, i, context_end
        
        # Look for common Manim error patterns
        error_patterns = [
            r'AttributeError:.*?\.(\w+)',  # Attribute errors
            r'NameError: name \'(\w+)\' is not defined',  # Undefined names
            r'TypeError:.*?\((.*?)\)'  # Type errors
        ]
        for pattern in error_patterns:
            match = re.search(pattern, error_msg)
            if match:
                keyword = match.group(1)
                for i, line in enumerate(lines):
                    if keyword in line:
                        start_line = max(0, i - 10)
                        end_line = min(len(lines), i + 10)
                        return '\n'.join(lines[start_line:end_line]), start_line, end_line
        
        # Fallback: return first 100 lines or entire code
        max_lines = min(100, len(lines))
        return '\n'.join(lines[:max_lines]), 0, max_lines
    
    def get_error_type(self, error_msg):
        """Extract the error type from the last part of the error message."""
        # Get the last line which usually contains the actual error
        lines = error_msg.strip().split('\n')
        last_line = lines[-1].strip()
        
        # Extract error type and message
        if ':' in last_line:
            error_type = last_line.split(':')[0].strip()
            error_details = ':'.join(last_line.split(':')[1:]).strip()
        else:
            error_type = last_line
            error_details = ""
        
        return error_type, error_details
    
    def apply_quick_fix(self, error_type, error_details, problematic_code):
        """Apply quick rule-based fixes for common Manim errors."""
        fixed_code = problematic_code
        
        if error_type == "AttributeError":
            # Common AttributeError fixes
            if "unexpected keyword argument" in error_details:
                # Remove unsupported arguments
                if "start_angle" in error_details:
                    fixed_code = re.sub(r',?\s*start_angle\s*=\s*[^,)]+', '', fixed_code)
                if "end_angle" in error_details:
                    fixed_code = re.sub(r',?\s*end_angle\s*=\s*[^,)]+', '', fixed_code)
                if "buff" in error_details and "move_to" in fixed_code:
                    fixed_code = re.sub(r'\.move_to\([^)]+,\s*buff\s*=\s*[^)]+\)', 
                                      lambda m: m.group(0).split(',')[0] + ')', fixed_code)
            
            # Fix deprecated mobject names
            if "SVGMobject" in error_details:
                fixed_code = fixed_code.replace("SVGMobject", "VMobject")
            if "Checkmark" in error_details:
                fixed_code = fixed_code.replace("Checkmark", "Text('✓')")
            if "Cross" in error_details:
                fixed_code = fixed_code.replace("Cross", "Text('✗')")
                
        elif error_type == "TypeError":
            # Fix Text() constructor issues
            if "Text" in error_details and "font_size" in fixed_code:
                # Ensure font_size is limited and add width
                fixed_code = re.sub(r'Text\([^)]+\)', 
                                  lambda m: self._fix_text_constructor(m.group(0)), fixed_code)
                                  
        elif error_type == "NameError":
            # Fix undefined names
            if "VGroup" in error_details:
                if "from manim import" in fixed_code:
                    fixed_code = fixed_code.replace("from manim import", "from manim import VGroup,")
                else:
                    fixed_code = "from manim import VGroup\n" + fixed_code
                    
        elif error_type == "ImportError" or error_type == "ModuleNotFoundError":
            # Fix import issues
            if "manim" in error_details:
                # Add missing imports
                missing_imports = ["Scene", "VGroup", "Text", "Rectangle", "Circle", "VMobject"]
                for imp in missing_imports:
                    if imp in fixed_code and f"import {imp}" not in fixed_code:
                        if "from manim import" in fixed_code:
                            fixed_code = fixed_code.replace("from manim import", f"from manim import {imp},")
                        else:
                            fixed_code = f"from manim import {imp}\n" + fixed_code
        
        return fixed_code
    
    def _fix_text_constructor(self, text_call):
        """Fix Text() constructor to ensure proper font_size and width."""
        # Extract existing parameters
        match = re.match(r'Text\(([^)]+)\)', text_call)
        if not match:
            return text_call
        
        params = match.group(1)
        
        # Fix font_size if present
        if "font_size=" in params:
            params = re.sub(r'font_size\s*=\s*(\d+)', 
                           lambda m: f"font_size={min(int(m.group(1)), 36)}", params)
        else:
            params += ", font_size=24"
        
        # Add width if not present``
        if "width=" not in params:
            params += ", width=7.0"
        
        return f"Text({params})"
    
    def fix_specific_error(self, error_msg, problematic_code, context_before="", context_after=""):
        """Fix specific error using quick rule-based approach first, fallback to LLM if needed."""
        error_type, error_details = self.get_error_type(error_msg)
        logging.info(f"Error type: {error_type}, Details: {error_details[:100]}...")
        
        # Try quick fix first
        quick_fixed = self.apply_quick_fix(error_type, error_details, problematic_code)
        
        # If quick fix made changes, return it
        if quick_fixed != problematic_code:
            logging.info("Applied quick rule-based fix")
            return quick_fixed
        
        # Fallback to LLM for complex cases (simplified prompt)
        logging.info("Falling back to LLM for complex error")
        prompt = f"""Fix this {error_type} in Manim code:
Error: {error_details}
Code:
```python
{problematic_code}
```
Fixed:
```python
"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "```python" in result:
                result = result.split("```python")[-1]
            if "```" in result:
                result = result.split("```")[0]
            return result.strip()
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            return problematic_code
    
    def fix_file_with_error(self, file_path, error_msg, return_fixed_code=False, overwrite_original=False):
        """Fix a file based on error message, optionally return fixed code."""
        logging.info(f"Fixing file: {file_path} for error: {error_msg}")
        file_path = Path(file_path)
        if not file_path.exists():
            logging.error(f"File not found: {file_path}")
            return None if not return_fixed_code else (None, None)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_code = f.read()
            logging.info(f"Read file: {file_path}")
        except Exception as e:
            logging.error(f"Error reading file: {e}")
            return None if not return_fixed_code else (None, None)
        
        problematic_section, start_line, end_line = self.extract_problematic_section(full_code, error_msg)
        lines = full_code.splitlines()
        context_before = '\n'.join(lines[:start_line]) if start_line > 0 else ""
        context_after = '\n'.join(lines[end_line:]) if end_line < len(lines) else ""
        
        logging.info(f"Identified problematic section: lines {start_line+1}-{end_line}")
        fixed_section = self.fix_specific_error(error_msg, problematic_section, context_before, context_after)
        
        if context_before and context_after:
            fixed_full_code = context_before + '\n' + fixed_section + '\n' + context_after
        elif context_before:
            fixed_full_code = context_before + '\n' + fixed_section
        elif context_after:
            fixed_full_code = fixed_section + '\n' + context_after
        else:
            fixed_full_code = fixed_section
        
        # Determine output strategy
        if overwrite_original:
            output_file = file_path  # Use the same file
        else:
            output_file = file_path.parent / (file_path.stem + "_fixed" + file_path.suffix)
        
        # Create temporary file for processing (different from output file)
        temp_file = file_path.parent / (file_path.stem + "_temp_processing" + file_path.suffix)
        
        try:
            logging.info(f"Fixed code preview: {fixed_full_code[:200]}...")
            # Write to the output file (either original or new file)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(fixed_full_code)
            logging.info(f"Fixed file saved as: {output_file}")
            
            # Also create temp file if cleanup is enabled (for any intermediate processing)
            if self.cleanup_fixed_files and not overwrite_original:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_full_code)
            
            if return_fixed_code:
                return fixed_full_code
            return str(output_file)
        except Exception as e:
            logging.error(f"Error saving fixed file: {e}")
            return None if not return_fixed_code else (None, None)
        finally:
            # Only cleanup the temporary processing file, NOT the output file
            if self.cleanup_fixed_files and temp_file.exists():
                try:
                    temp_file.unlink()
                    logging.info(f"Removed temporary processing file: {temp_file}")
                except Exception as e:
                    logging.warning(f"Could not remove temporary processing file {temp_file}: {e}")
                    
if __name__ == "__main__":
    fixer = TargetedCodeFixer()
    # Example usage
    error_message = "Mobject.__init__() got an unexpected keyword argument 'start_angle''"
    file_path = "test.py"
    fixed_file = fixer.fix_file_with_error(file_path, error_message, return_fixed_code=True)
    if fixed_file:
        logging.info(f"Fixed code saved to: {fixed_file}")
    else:
        logging.error("Failed to fix the code.")