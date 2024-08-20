data=[]
plot_data=[]

with open('time.txt', 'r') as file:
    content = file.read()
    data.append(content.split("\n"))

t=data[0][0].split(" ")

x=float(t[1])

with open('final_time.txt', 'a') as file:
    file.write(str(x)+' ')