// Obtener el botón
const scrollToTopBtn = document.getElementById('scrollToTopBtn');

// Mostrar el botón cuando se desplaza hacia abajo
window.onscroll = function() {
    if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
        scrollToTopBtn.style.display = 'block';
    } else {
        scrollToTopBtn.style.display = 'none';
    }
};

// Hacer que el botón funcione
scrollToTopBtn.onclick = function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
};

function focusInfo() {
    console.log("Haciendo foco en Economía...");
    var InfoDiv = document.getElementById('Info');
    if (InfoDiv) {
        InfoDiv.scrollIntoView({ behavior: 'smooth' });
    }
}


function focusEstadisticas() {
    console.log("Haciendo foco en Economía...");
    var EstadisticasDiv = document.getElementById('estadisticas');
    if (EstadisticasDiv) {
        EstadisticasDiv.scrollIntoView({ behavior: 'smooth' });
    }
}