import ast
import os
import sys

if len(sys.argv) != 2:
    print("Please input python project directory")
    exit(0)
# Define the path to your project directory
project_directory = sys.argv[1]

# Set to store imported libraries
imported_libraries = set()


# Function to traverse the AST and extract imports
def extract_imports(node):
    if isinstance(node, ast.Import):
        for n in node.names:
            imported_libraries.add(n.name)
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            for n in node.names:
                imported_libraries.add(node.module + '.' + n.name)


# Iterate through Python files in the project directory
for root, _, files in os.walk(project_directory):
    for file in files:
        if file.endswith('.py'):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                try:
                    tree = ast.parse(f.read(), filename=file)
                    for node in ast.walk(tree):
                        extract_imports(node)
                except SyntaxError as e:
                    print(f"Syntax error in {file}: {e}")

libraries = set()
# Print the list of imported libraries
for library in sorted(imported_libraries):
    # Get base library only
    library = library.split(".")[0]
    libraries.add(library)
libraries = sorted(libraries, key=lambda x:x, reverse=False)
for library in libraries:
    print(library)
