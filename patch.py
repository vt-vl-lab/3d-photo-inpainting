print("Patching files")

search_text = "yaml.load"
replace_text = "yaml.full_load"

def double_quote(word):
    return '"%s"' % word

def SearchAndReplace(filename, search_text, replace_text):
  # Opening the file in read only mode using the open() function
  with open(filename, "r") as file:
    # Reading the content of the file and putting them into the data variable
    data = file.read()
    # Searching and replacing the text
    data = data.replace(search_text, replace_text)
  
  # Opening our text file in write only mode to write the replaced content
  with open(filename, "w") as file:
    # Writing the replaced data in our text file
    file.write(data)
  
  # Printing Text replaced
  print(double_quote(search_text) + " was replaced with " + double_quote(replace_text) + ".")

SearchAndReplace("main.py", "yaml.load", "yaml.full_load")