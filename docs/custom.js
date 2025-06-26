var experimentsLink = Array.from(document.querySelectorAll('.md-tabs__link'))
    .find(link => link.textContent.trim() === 'Publications');
if (experimentsLink) {
    experimentsLink.classList.add('blurry-tab');
}