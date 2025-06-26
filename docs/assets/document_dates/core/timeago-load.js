if (typeof timeago !== 'undefined') {
    document.querySelectorAll('.document-dates-plugin time').forEach(timeElement => {
        timeElement.textContent = timeago.format(timeElement.getAttribute('datetime'), timeElement.getAttribute('locale'));
    });
}