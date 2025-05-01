// Console log to confirm that mood_graph.js has been loaded successfully
console.log("✅ mood_graph.js loaded!");

// Wait until the full page DOM has loaded before running the script
document.addEventListener("DOMContentLoaded", async () => {
    // Get the context for the mood chart canvas
    const ctx = document.getElementById("moodChart").getContext("2d");
    // Get the time filter dropdown and refresh button elements
    const timeFilter = document.getElementById("timeFilter");
    const refreshBtn = document.getElementById("refreshBtn");

    let allData = null; // Variable to store the complete dataset of mood logs

    // Mapping moods to numerical values for plotting purposes
    const moodMap = { "sad": 0, "anxious": 1, "neutral": 2, "happy": 3, "stressed": 4 };

    // Function to fetch mood log data from the server
    const fetchData = async () => {
        try {
            const res = await fetch("/api/mood-logs");
            const data = await res.json();
            allData = data; // Store the fetched data
            updateChart("all"); // Initially display all data
        } catch (err) {
            console.error("❌ Failed to load mood data:", err);
        }
    };

    // Function to update the chart based on the selected time filter
    const updateChart = (filter) => {
        const now = new Date(); // Current date and time
        let filteredTimestamps = [];
        let filteredMoods = [];
        let filteredSentiments = [];

        // Filter mood entries based on the selected timeframe
        allData.timestamps.forEach((timestamp, index) => {
            const logTime = new Date(timestamp);
            const daysDiff = (now - logTime) / (1000 * 60 * 60 * 24); // Calculate difference in days

            if (
                filter === "all" ||
                (filter === "7" && daysDiff <= 7) ||
                (filter === "30" && daysDiff <= 30)
            ) {
                filteredTimestamps.push(timestamp);
                filteredMoods.push(allData.moods[index]);
                filteredSentiments.push(allData.sentiments[index]);
            }
        });

        // Update the chart labels and datasets
        myChart.data.labels = filteredTimestamps;
        myChart.data.datasets[0].data = filteredMoods.map(m => moodMap[m] ?? 2); // Default to 'neutral' if mood is unknown
        myChart.data.datasets[1].data = filteredSentiments.map(s => 
            s === "positive" ? 1 : (s === "negative" ? -1 : 0)
        );
        myChart.update(); // Refresh the chart display
    };

    // Initialise the Chart.js line chart
    const myChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [], // X-axis labels (timestamps)
            datasets: [
                {
                    label: "Mood", // Dataset for mood levels
                    data: [],
                    borderColor: "#e91e63",
                    backgroundColor: "rgba(233, 30, 99, 0.2)",
                    tension: 0.3, // Smoother curve
                    fill: true
                },
                {
                    label: "Sentiment", // Dataset for sentiment values
                    data: [],
                    borderColor: "#3f51b5",
                    backgroundColor: "rgba(63, 81, 181, 0.2)",
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true, // Make the chart adapt to different screen sizes
            scales: {
                y: {
                    beginAtZero: true, // Always start the y-axis at 0
                    ticks: {
                        stepSize: 1, // Step by 1 unit
                        callback: (value) => {
                            // Convert numeric mood values back into labels
                            const moodLabels = ["Sad", "Anxious", "Neutral", "Happy", "Stressed"];
                            return moodLabels[value] ?? value;
                        }
                    }
                }
            }
        }
    });

    // Listen for changes on the time filter dropdown and update the chart accordingly
    timeFilter.addEventListener("change", () => updateChart(timeFilter.value));

    // Listen for clicks on the refresh button and reload the page
    refreshBtn.addEventListener("click", () => location.reload());

    // Load the initial data when the page loads
    fetchData();
});
