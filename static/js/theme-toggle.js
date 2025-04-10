document.addEventListener("DOMContentLoaded", () => {
    const toggleBtn = document.getElementById("toggle-theme");
    const isDark = localStorage.getItem("theme") === "dark";

    if (isDark) {
        document.body.classList.add("dark");
        toggleBtn.textContent = "☀️";
    }

    toggleBtn.addEventListener("click", () => {
        document.body.classList.toggle("dark");
        const darkMode = document.body.classList.contains("dark");
        toggleBtn.textContent = darkMode ? "☀️" : "🌙";
        localStorage.setItem("theme", darkMode ? "dark" : "light");
    });
});
