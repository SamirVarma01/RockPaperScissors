document.addEventListener("DOMContentLoaded", () => {
    const startGameBtn = document.getElementById("start-game");
    const countdownElement = document.getElementById("countdown");
    const playerChoiceElement = document.getElementById("player-choice");
    const botChoiceElement = document.getElementById("bot-choice");
    const resultElement = document.getElementById("result");
    const videoFeed = document.getElementById("video-feed");
        
    videoFeed.onerror = (error) => {
        console.error("Video feed error:", error);
        countdownElement.textContent = "Failed to load video feed. Check console for details.";
    };

    videoFeed.onload = () => {
        console.log("Video feed loaded successfully");
    };

    startGameBtn.addEventListener("click", async () => {
        try {
            // Start countdown
            for (let i = 3; i > 0; i--) {
                countdownElement.textContent = `Game starts in ${i}...`;
                await new Promise(resolve => setTimeout(resolve, 1000));
            }
            countdownElement.textContent = "Go!";

            // Send a POST request to the backend
            const response = await fetch("/play", { method: "POST" });

            // Ensure the response is OK
            if (!response.ok) {
                throw new Error("Failed to connect to server.");
            }

            const data = await response.json();

            // Update the UI with results
            if (data.error) {
                countdownElement.textContent = "Error occurred!";
                playerChoiceElement.textContent = "-";
                botChoiceElement.textContent = "-";
                resultElement.textContent = data.error;
            } else {
                countdownElement.textContent = "Game Over!";
                playerChoiceElement.textContent = data.player_choice;
                botChoiceElement.textContent = data.bot_choice;
                resultElement.textContent = data.result;
            }
        } catch (error) {
            // Handle any unexpected errors
            countdownElement.textContent = "An error occurred!";
            console.error("Error:", error);
        }
    async function checkCameraPermission() {
        try {
            await navigator.mediaDevices.getUserMedia({ video: true });
            console.log("Camera permission granted");
            return true;
        } catch (err) {
            console.error("Camera permission error:", err);
            return false;
        }
     }
    checkCameraPermission();
    });
});