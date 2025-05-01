// scripts.js

// Add event listeners to enhance user experience
document.addEventListener('DOMContentLoaded', () => {
    // Confirm that JavaScript is connected
    console.log('JavaScript is connected and ready!');

    // --- Logout Confirmation ---
    // Add a confirmation pop-up when the user attempts to log out
    const logoutLink = document.querySelector('a[href="/logout"]');
    if (logoutLink) {
        logoutLink.addEventListener('click', (event) => {
            const confirmLogout = confirm("Are you sure you want to log out?");
            if (!confirmLogout) {
                event.preventDefault(); // Cancel logout if the user selects 'Cancel'
            }
        });
    }

    // --- Emotion Input Validation on Dashboard ---
    // Ensure that the emotion input is not empty before form submission
    const emotionForm = document.querySelector('form[action="/dashboard"]');
    if (emotionForm) {
        emotionForm.addEventListener('submit', (event) => {
            const emotionInput = document.querySelector('input[name="emotion"]');
            if (!emotionInput.value.trim()) {
                alert("Please enter your emotion before submitting.");
                event.preventDefault(); // Stop the form from submitting if the field is empty
            }
        });
    }
});
