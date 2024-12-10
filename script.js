document.getElementById("checkButton").addEventListener("click", async () => {
    const emailContent = document.getElementById("emailContent").value;
    const resultField = document.getElementById("result");
  
    if (emailContent.trim() === "") {
      resultField.textContent = "Please enter title of email.";
      return;
    }
  
    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: emailContent })
      });
  
      if (response.ok) {
        const data = await response.json();
        const isPhishing = data.prediction === 1;
        resultField.textContent = isPhishing ? "Email không an toàn,chú ý nhé !" : "Email an toàn ";
      } else {
        resultField.textContent = "Error sending request to server.";
      }
    } catch (error) {
      resultField.textContent = "Unable to connect to server.";
    }
  });
  