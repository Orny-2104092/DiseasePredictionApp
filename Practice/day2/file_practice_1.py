name = input("Enter your name: ")
age = input("Enter your age: ")

with open("user_data.txt", "w") as file:
    file.write(f"Name: {name}\n")
    file.write(f"Age: {age}\n")

print("✅ Data saved to user_data.txt successfully!")
