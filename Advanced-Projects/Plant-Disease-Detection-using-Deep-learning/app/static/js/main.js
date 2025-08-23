document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('imagefile');

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('border-indigo-500');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('border-indigo-500');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('border-indigo-500');
        fileInput.files = e.dataTransfer.files;
        dropzone.querySelector('p').textContent = fileInput.files[0].name || 'Select an image';
    });

    fileInput.addEventListener('change', () => {
        dropzone.querySelector('p').textContent = fileInput.files[0].name || 'Select an image';
    });
});