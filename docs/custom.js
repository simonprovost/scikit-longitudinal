var communityLink = Array.from(document.querySelectorAll('.md-tabs__link'))
    .find(link => link.textContent.trim() === 'Community Hub');
if (communityLink) {
    communityLink.classList.add('blurry-tab');
}