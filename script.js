// Visitor Tracking
function logVisit() {
    const page = window.location.pathname.split("/").pop() || "index.html";
    const timestamp = new Date().toISOString();

    let visits = JSON.parse(localStorage.getItem('website_visits') || '[]');
    visits.push({ page, timestamp });

    // Keep only last 1000 logs to prevent storage bloat
    if (visits.length > 1000) visits = visits.slice(-1000);

    localStorage.setItem('website_visits', JSON.stringify(visits));
    console.log(`Visit logged: ${page} at ${timestamp}`);
}

// Initialize Feather icons and logging
document.addEventListener("DOMContentLoaded", () => {
    feather.replace();
    logVisit();
});

// Simple page transiton animation helper
window.addEventListener('pageshow', (event) => {
    if (event.persisted) {
        // Clear animation cache if loaded from bfcache
        document.body.classList.remove('animate-in');
        void document.body.offsetWidth; // trigger reflow
        document.body.classList.add('animate-in');
    }
});
