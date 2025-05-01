// profile.js - JavaScript for the user profile page

// --- Mood Chart and CSV Export Logic ---
// Wait for the DOM to fully load before executing code
document.addEventListener("DOMContentLoaded", function () {
  // Find the element containing mood data (embedded in the HTML)
  const moodDataEl = document.getElementById("mood-data");
  
  // Only proceed if the mood data element exists
  if (moodDataEl) {
    // Parse the embedded JSON mood data from the element
    const data = JSON.parse(moodDataEl.textContent);
    const moods = data.moods;             // Array of mood values
    const sentiments = data.sentiments;   // Array of sentiment values
    const timestamps = data.timestamps;   // Array of timestamp strings

    // CSV Export Button functionality
    const exportBtn = document.getElementById("exportCsv");
    if (exportBtn) {
      // Add click event listener to the export button
      exportBtn.addEventListener("click", () => {
        // Construct the CSV content with a header row and data rows
        const csvRows = [
          ["Timestamp", "Mood", "Sentiment"],  // Header row
          ...timestamps.map((t, i) => [t, moods[i], sentiments[i]])  // Data rows
        ];
        // Join rows and columns to create CSV format
        const csv = csvRows.map(row => row.join(",")).join("\n");

        // Create a downloadable CSV file using the Blob API
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        // Create a temporary link element to trigger download
        const a = document.createElement("a");
        a.href = url;
        a.download = "mood_history.csv"; // Set file name
        a.click();  // Simulate click to start download
        URL.revokeObjectURL(url); // Clean up the object URL to prevent memory leaks
      });
    }

    // Mood Chart drawing using Chart.js
    // Get canvas context for drawing the chart, using optional chaining (?.) for safety
    const ctx = document.getElementById("profileMoodChart")?.getContext("2d");
    if (ctx) {
      // Create a new Chart.js line chart
      new Chart(ctx, {
        type: "line",  // Chart type
        data: {
          labels: timestamps, // X-axis: timestamps
          datasets: [
            {
              label: "Mood",  // First dataset - mood line
              // Map mood strings to numeric values for chart display
              // 0=sad, 1=anxious, 2=neutral, 3=happy, 4=stressed
              // Default to 2 (neutral) if mood not recognized
              data: moods.map(m => ({ "sad": 0, "anxious": 1, "neutral": 2, "happy": 3, "stressed": 4 }[m] ?? 2)),
              borderColor: "#e91e63",  // Pink line
              backgroundColor: "rgba(233, 30, 99, 0.2)",  // Light pink fill
              tension: 0.3,  // Line smoothing
              fill: true  // Fill area under the line
            },
            {
              label: "Sentiment",  // Second dataset - sentiment line
              // Map sentiment strings to numeric values: 1=positive, 0=neutral, -1=negative
              data: sentiments.map(s => s === "positive" ? 1 : (s === "negative" ? -1 : 0)),
              borderColor: "#3f51b5",  // Blue line
              backgroundColor: "rgba(63, 81, 181, 0.2)",  // Light blue fill
              tension: 0.3,  // Line smoothing
              fill: true  // Fill area under the line
            }
          ]
        },
        options: {
          responsive: true, // Chart resizes with window
          scales: {
            y: {
              beginAtZero: true, // Start y-axis from zero
              ticks: {
                stepSize: 1,  // Increment by 1
                // Convert numeric y-axis values back to mood labels for display
                callback: value => ["Sad", "Anxious", "Neutral", "Happy", "Stressed"][value] || value
              }
            }
          }
        }
      });
    }
  }

  // --- AJAX Toggle Recommendation Completion ---
  // Find all progress buttons with data-item-id attribute
  document.querySelectorAll("button[data-item-id]").forEach(button => {
    // Add click event listener to each button
    button.addEventListener("click", () => {
      // Get the item ID from the button's data attribute
      const itemId = button.dataset.itemId;

      // Send AJAX POST request to toggle completion status
      fetch("/toggle_progress_ajax", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_id: itemId })
      })
      .then(res => res.json())  // Parse JSON response
      .then(data => {
        // If the server operation was successful
        if (data.success) {
          // Update button appearance based on new completion status
          button.classList.toggle("btn-outline-secondary");  // Toggle grey outline style
          button.classList.toggle("btn-success");  // Toggle green success style
          button.innerText = data.completed ? "✓ Completed" : "Mark as Done";  // Update button text

          // Show a toast notification with the update message
          const toast = document.getElementById("liveToast");
          const toastMsg = document.getElementById("toast-message");
          toastMsg.innerText = data.message;  // Set toast message text
          new bootstrap.Toast(toast).show();  // Display the toast
        }
      })
      .catch(console.error); // Log any errors to the console
    });
  });
});
