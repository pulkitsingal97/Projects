import matplotlib.pyplot as plt

data=[]
with open('final_time.txt', 'r') as file:
    content = file.read()
    data.append(content)

data=data[0]
data=data.split(" ")

for i in range(len(data)-1):
    data[i]=float(data[i])

thres = [5, 10, 25, 50, 95]

plt.plot(thres, data[0:5], color='blue', marker='o',label='gSpan')
plt.plot(thres, data[5:10], color='orange', marker='o', label='FSG')
plt.plot(thres, data[10:15], color='green', marker='o', label='gaston')
plt.xlabel('minSup (%)')
plt.ylabel('Running Time (s)')
plt.title('Running Time vs minSup')
plt.legend()
plt.savefig("q1_AIB232064.png")

with open('final_time.txt', 'w') as file:
	pass