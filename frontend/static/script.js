document.getElementById("chat-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const question = document.getElementById("question").value;
    const responseDiv = document.getElementById("response");
    responseDiv.innerHTML = "Thinking...";

    try {
        const res = await fetch("http://127.0.0.1:8000/mcp-query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        const data = await res.json();
        if (data.response) {
            responseDiv.innerHTML = `<strong>Answer:</strong> ${data.response}`;
        } else {
            responseDiv.innerHTML = `<span style="color:red;">Error: ${data.error}</span>`;
        }
    } catch (error) {
        responseDiv.innerHTML = `<span style="color:red;">Failed to connect to the server.</span>`;
    }
});
