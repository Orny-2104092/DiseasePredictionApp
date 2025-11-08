#task1:
name=input("Enter your name: ")
age=int(input("Enter your age: "))
print(f"hello {name}, you are {age} years old.")

#task2:
temp=float(input("Enter temperature in Celsius: "))
if temp>38:
    print("you have a fever")
elif temp>36.5:
    print("your temperature is normal")
else:
    print("Low temperature detected")

#task3:
symptoms=[ "cough" ,"fever","headache"]

for s in symptoms:
    print(s)
