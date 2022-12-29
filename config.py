FPS = 30 #@param {type:"integer"}
Frames = 180 #@param {type:"integer"}
Image_Input_Folder = "./KenBurns/Input" #@param {type:"string"}
Video_Output_Folder = "./KenBurns/Output" #@param {type:"string"}

def GetLineNumber(filename, searchtext):
  with open(filename) as myFile:
    for num, line in enumerate(myFile, 0):
        if searchtext in line:
            myFile.close()
            return num
            #print('found at line:', num)

def ReplaceLine(file_name, line_num, text):
    text += "\n"
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

# FPS change
filename = "argument.yml"
searchtext = "fps: "
newtext = searchtext + str(FPS)
linenum = GetLineNumber(filename, searchtext)
print (searchtext + " is on line " + str(linenum))
ReplaceLine(filename, linenum, newtext)

# Frames change
filename = "argument.yml"
searchtext = "num_frames: "
newtext = searchtext + str(Frames)
linenum = GetLineNumber(filename, searchtext)
print (searchtext + " is on line " + str(linenum))
ReplaceLine(filename, linenum, newtext)

# Input folder change
filename = "argument.yml"
searchtext = "src_folder: "
newtext = searchtext + Image_Input_Folder
linenum = GetLineNumber(filename, searchtext)
print (searchtext + " is on line " + str(linenum))
ReplaceLine(filename, linenum, newtext)

# Output folder change
filename = "argument.yml"
searchtext = "video_folder: "
linenum = GetLineNumber(filename, searchtext)
print (searchtext + " is on line " + str(linenum))
newtext = searchtext + Video_Output_Folder
ReplaceLine(filename, linenum, newtext)