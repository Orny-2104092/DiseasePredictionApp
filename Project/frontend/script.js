async function predictDisease() {
    const userInput = document.getElementById("symptomInput").value;

    if (userInput.trim() === "") {
        alert("Please enter symptoms!");
        return;
    }

    const url = "http://127.0.0.1:8000/predict_text";

    const formData = new FormData();
    formData.append("user_input", userInput);

    try {
        const response = await fetch(url, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            alert("API Error — check console!");
            console.log(await response.text());
            return;
        }

        const data = await response.json();
        console.log("API Response:", data);

        // Show predicted disease
        const disease = data["predicted_disease"];
        document.getElementById("disease").innerText = disease;

        // Now fetch description & precautions
        await loadDiseaseInfo(disease);

        document.getElementById("resultBox").classList.remove("hidden");

    } catch (error) {
        alert("API Error — check console!");
        console.error(error);
    }
}


async function loadDiseaseInfo(disease) {
    try {
        const res = await fetch(`http://127.0.0.1:8000/get_details?disease=${encodeURIComponent(disease)}`);

        if (!res.ok) {
            console.error("Backend Error:", await res.text());
            return;
        }

        const data = await res.json();
        console.log("Details:", data);

        document.getElementById("description").innerText = data.description;

        const precautionList = document.getElementById("precautions");
        precautionList.innerHTML = "";

        data.precautions.forEach(p => {
            const li = document.createElement("li");
            li.innerText = p;
            precautionList.appendChild(li);
        });

    } catch (error) {
        console.error("Fetch failed:", error);
    }
}
