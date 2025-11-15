async function predictDisease() {
    const userInput = document.getElementById("symptomInput").value;

    if (userInput.trim() === "") {
        alert("Please enter symptoms!");
        return;
    }

    // API URL (FastAPI backend)
    const url = "http://127.0.0.1:8000/predict_text";

    const formData = new FormData();
    formData.append("user_input", userInput);

    const response = await fetch(url, {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    // Show prediction on UI
    document.getElementById("disease").innerText = data["Predicted Disease"];
    document.getElementById("description").innerText = data["Description"];

    const precautionList = document.getElementById("precautions");
    precautionList.innerHTML = "";

    if (data["Precautions"].length > 0) {
        data["Precautions"].forEach(p => {
            const li = document.createElement("li");
            li.textContent = p;
            precautionList.appendChild(li);
        });
    }

    document.getElementById("resultBox").classList.remove("hidden");
}
