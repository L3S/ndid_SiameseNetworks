template = """embeddings {
  tensor_name: "REPLACE_ME"
  metadata_path: "REPLACE_ME/metadata.tsv"
  tensor_path: "REPLACE_ME/values.tsv"
}
"""

# Using readlines()
inFile = open('projector_config.txt', 'r')
Lines = inFile.readlines()

outFile = open("projector_config.pbtxt", "w")

count = 0
# Strips the newline character
for line in Lines:
    count += 1
    outFile.write(template.replace("REPLACE_ME", line.strip()))

outFile.close()
