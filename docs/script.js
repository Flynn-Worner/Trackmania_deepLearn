document.addEventListener('DOMContentLoaded', () => {
    
    // --- Navigation Background on Scroll ---
    const nav = document.querySelector('nav');
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });

    // --- Copy to Clipboard Functionality ---
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', () => {
            const codeToCopy = button.getAttribute('data-clipboard');
            
            navigator.clipboard.writeText(codeToCopy).then(() => {
                const originalText = button.innerText;
                button.innerText = 'Copied!';
                button.style.color = 'var(--accent-primary)';
                
                setTimeout(() => {
                    button.innerText = originalText;
                    button.style.color = '';
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy text: ', err);
            });
        });
    });

    // --- Intersection Observer for Scroll Animations ---
    const animationObserverOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.15
    };

    const animationObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Optional: stop observing once it's visible so it only animates once
                observer.unobserve(entry.target);
            }
        });
    }, animationObserverOptions);

    const animatedElements = document.querySelectorAll('.fade-in, .fade-in-up, .fade-in-left, .fade-in-right');
    animatedElements.forEach(el => {
        animationObserver.observe(el);
    });
    
});
