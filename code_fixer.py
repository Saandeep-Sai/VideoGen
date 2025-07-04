#!/usr/bin/env python3
"""
Targeted Code Fixer for Manim Scripts
Fixes specific errors in Python/Manim code files efficiently
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import sys
from tqdm import tqdm
import os
import re
from pathlib import Path

class TargetedCodeFixer:
    def __init__(self, model_id="deepseek-ai/deepseek-coder-1.3B-instruct"):
        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Initialize tokenizer and model"""
        print(f"🚀 Loading {self.model_id}")
        print(f"📱 Device: {self.device}")
        
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def extract_problematic_section(self, full_code, error_msg):
        """Extract the problematic section based on error message"""
        lines = full_code.splitlines()
        
        # Look for line numbers in error message
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            error_line = int(line_match.group(1)) - 1  # Convert to 0-based indexing
            start_line = max(0, error_line - 10)
            end_line = min(len(lines), error_line + 10)
            return '\n'.join(lines[start_line:end_line]), start_line, end_line
        
        # Look for function/class names in error
        for i, line in enumerate(lines):
            if 'def ' in line or 'class ' in line:
                # Extract function/class context
                indent_level = len(line) - len(line.lstrip())
                context_end = i + 1
                
                # Find end of function/class
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) <= indent_level:
                        if not lines[j].strip().startswith(('"""', "'''")):
                            context_end = j
                            break
                    context_end = j + 1
                
                context = '\n'.join(lines[i:context_end])
                if any(keyword in error_msg.lower() for keyword in ['vmobject', 'mobject', 'vgroup']):
                    return context, i, context_end
        
        # If no specific section found, return a reasonable chunk
        return '\n'.join(lines[:50]), 0, 50
    
    def fix_specific_error(self, error_msg, problematic_code, context_before="", context_after=""):
        """Fix specific error in a targeted way"""
        
        prompt = f"""### Error Analysis and Fix
You are a Python Manim code expert. Fix the specific error in the given code section.

**Error Message:**
{error_msg}

**Problematic Code:**
```python
{problematic_code}
```

**Context Before:**
```python
{context_before[-200:] if context_before else "# No context before"}
```

**Context After:**
```python
{context_after[:200] if context_after else "# No context after"}
```

### Instructions:
1. Identify the exact cause of the error
2. Provide ONLY the corrected version of the problematic code section
3. Ensure compatibility with Manim v0.19.0
4. Focus on the specific error - don't rewrite unrelated code
5. Maintain the same functionality and structure

### Fixed Code:
```python
"""
        
        print(f"🔍 Analyzing error: {error_msg[:100]}...")
        print(f"📝 Code section: {len(problematic_code)} characters")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the fixed code
            if "### Fixed Code:" in result:
                result = result.split("### Fixed Code:")[-1].strip()
            
            # Clean up code blocks
            if "```python" in result:
                result = result.split("```python")[-1]
            if "```" in result:
                result = result.split("```")[0]
            
            return result.strip()
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return problematic_code
    
    def fix_file_with_error(self, file_path, error_msg):
        """Fix a file based on error message"""
        
        # Convert to Path object for better handling
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            print(f"📁 Absolute path: {file_path.absolute()}")
            return None
        
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_code = f.read()
            print(f"✅ Successfully read file: {file_path}")
            print(f"📄 File size: {len(full_code)} characters")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            print(f"📁 Attempted path: {file_path.absolute()}")
            return None
        
        # Extract problematic section
        problematic_section, start_line, end_line = self.extract_problematic_section(full_code, error_msg)
        
        lines = full_code.splitlines()
        context_before = '\n'.join(lines[:start_line]) if start_line > 0 else ""
        context_after = '\n'.join(lines[end_line:]) if end_line < len(lines) else ""
        
        print(f"📍 Identified problematic section: lines {start_line+1}-{end_line}")
        print(f"🔧 Fixing section...")
        
        # Fix the specific error
        fixed_section = self.fix_specific_error(error_msg, problematic_section, context_before, context_after)
        
        # Reconstruct the file
        if context_before and context_after:
            fixed_full_code = context_before + '\n' + fixed_section + '\n' + context_after
        elif context_before:
            fixed_full_code = context_before + '\n' + fixed_section
        elif context_after:
            fixed_full_code = fixed_section + '\n' + context_after
        else:
            fixed_full_code = fixed_section
        
        # Save the fixed file
        output_file = file_path.parent / (file_path.stem + '_fixed' + file_path.suffix)
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(fixed_full_code)
            print(f"✅ Fixed file saved as: {output_file}")
            return str(output_file)
        except Exception as e:
            print(f"❌ Error saving fixed file: {e}")
            return None
    
    def quick_fix(self, code_snippet, error_msg):
        """Quick fix for small code snippets"""
        print(f"🔧 Quick fixing code snippet...")
        return self.fix_specific_error(error_msg, code_snippet)

def main():
    """Main function to run the code fixer"""
    print("🚀 Targeted Code Fixer - Interactive Mode")
    print("=" * 60)
    
    # Initialize the fixer
    fixer = TargetedCodeFixer()
    
    while True:
        print("\n🔧 Choose an option:")
        print("1. Fix a file with error message")
        print("2. Quick fix code snippet")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n📁 File Fix Mode")
            file_path = input("Enter the file path: ").strip()
            
            # Remove quotes if present
            if file_path.startswith('"') and file_path.endswith('"'):
                file_path = file_path[1:-1]
            if file_path.startswith("'") and file_path.endswith("'"):
                file_path = file_path[1:-1]
            
            # Handle raw strings on Windows
            if '\\' in file_path:
                file_path = file_path.replace('\\\\', '\\')
            
            print(f"📄 Using file path: {file_path}")
            
            error_msg = input("Enter the error message: ").strip()
            
            if file_path and error_msg:
                result = fixer.fix_file_with_error(file_path, error_msg)
                if result:
                    print(f"🎉 Success! Fixed file created: {result}")
                else:
                    print("❌ Failed to fix the file")
            else:
                print("❌ Please provide both file path and error message")
        
        elif choice == '2':
            print("\n🔧 Quick Fix Mode")
            print("Enter your code snippet (press Enter twice to finish):")
            code_lines = []
            while True:
                line = input()
                if line == "" and len(code_lines) > 0 and code_lines[-1] == "":
                    break
                code_lines.append(line)
            
            code_snippet = '\n'.join(code_lines[:-1])  # Remove last empty line
            
            error_msg = input("Enter the error message: ").strip()
            
            if code_snippet and error_msg:
                fixed_code = fixer.quick_fix(code_snippet, error_msg)
                print("\n📋 Fixed code:")
                print("-" * 40)
                print(fixed_code)
                print("-" * 40)
            else:
                print("❌ Please provide both code snippet and error message")
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()