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

// Scroll Motion Background Shift
let lastScrollTop = 0;
let scrollTimeout;

window.addEventListener('scroll', () => {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    const distance = Math.abs(scrollTop - lastScrollTop);
    lastScrollTop = scrollTop;

    // Calculate speed factor (0 to 1) based on distance per frame
    // normalized to about 100px per scroll event as "max speed"
    const maxSpeed = 100;
    const speedFactor = Math.min(distance / maxSpeed, 1);

    // Shift background proportional to speed
    const bgColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-color').trim();
    const motionColor = getComputedStyle(document.documentElement).getPropertyValue('--bg-motion-color').trim();

    // Interpolate colors (simple approach using composite variable)
    document.body.style.backgroundColor = `color-mix(in srgb, ${motionColor} ${speedFactor * 100}%, ${bgColor})`;

    // Reset background color when scrolling stops
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
        document.body.style.backgroundColor = '';
    }, 150);
}, { passive: true });

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
