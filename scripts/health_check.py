import sys
import os
import importlib.util
from dotenv import load_dotenv

def check_package(package_name: str) -> bool:
    """Check if a package is installed and accessible."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ Missing package: {package_name}")
        return False
    print(f"✅ Found package: {package_name}")
    return True

def main():
    print("--- CPU Scheduling RL Platform Health Check ---\n")
    
    # Check Python version
    print(f"Python Version: {sys.version.split(' ')[0]}")
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required.")
        sys.exit(1)
        
    # Check dependencies
    print("\nChecking Dependencies:")
    packages = [
        "gymnasium", "stable_baselines3", "torch", "numpy", 
        "pandas", "matplotlib", "tabulate", "dotenv", "groq", "pytest"
    ]
    all_passed = True
    for pkg in packages:
        if not check_package(pkg):
            all_passed = False
            
    # Check Environment Variables
    print("\nChecking Environment Variables:")
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key != "gsk_your_api_key_here":
        print("✅ GROQ_API_KEY is configured.")
    else:
        print("⚠️ GROQ_API_KEY is not set or using default.")
        print("  (LLM features will automatically run in mock mode later, so this is fine for now)")
        
    # Final result
    if all_passed:
        print("\n🚀 Environment setup is complete and healthy!")
    else:
        print("\n❌ Environment setup failed. Please check the missing packages.")
        sys.exit(1)

if __name__ == "__main__":
    main()