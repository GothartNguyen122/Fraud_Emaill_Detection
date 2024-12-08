document.getElementById("checkButton").addEventListener("click", async () => {
    const emailContent = document.getElementById("emailContent").value;
    const resultField = document.getElementById("result");
  
    if (emailContent.trim() === "") {
      resultField.textContent = "Vui lòng nhập nội dung email.";
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
        resultField.textContent = isPhishing ? "Email này có khả năng là giả mạo!" : "Email này có vẻ an toàn.";
      } else {
        resultField.textContent = "Lỗi khi gửi yêu cầu đến server.";
      }
    } catch (error) {
      resultField.textContent = "Không thể kết nối tới server.";
    }
  });
  