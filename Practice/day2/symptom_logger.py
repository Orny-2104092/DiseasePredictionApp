print("=== Symptom Logger ===")

symptom1 = input("Enter symptom 1: ")
symptom2 = input("Enter symptom 2: ")
symptom3 = input("Enter symptom 3: ")

# Save to file
with open("symptoms.txt", "w") as f:
    f.write(symptom1 + "\n")
    f.write(symptom2 + "\n")
    f.write(symptom3 + "\n")

print("\n✅ Symptoms saved successfully!")

# Read back
print("\n📄 Saved Symptoms:")
with open("symptoms.txt", "r") as f:
    for line in f:
        print("-", line.strip())
