def savePredictionsToFile(filename, y, classes):
    f = open(filename, "a")
    f.write("Id,Category\n")
    for i, label in enumerate(y):
        output = str(i) + "," + classes[label] + "\n"
        f.write(output)
