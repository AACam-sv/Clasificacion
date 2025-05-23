// Cargar el modelo
let model;
async function loadModel() {
    model = await tf.loadLayersModel('/modelo_tfjs/model.json');
    console.log('Modelo cargado');
    document.getElementById('predictButton').disabled = false;
}

// Manejar la carga de la imagen
const imageUpload = document.getElementById('imageUpload');
const preview = document.getElementById('preview');
const predictButton = document.getElementById('predictButton');
const result = document.getElementById('result');

imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            predictButton.disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

// Realizar la predicción
predictButton.addEventListener('click', async () => {
    if (!model) {
        result.textContent = 'El modelo aún no está cargado';
        return;
    }

    const img = document.getElementById('preview');
    let tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([300, 300]) // Redimensionar a 300x300
        .toFloat()
        .div(tf.scalar(255)) // Normalizar a [0,1]
        .expandDims(); // Añadir dimensión de lote

    const prediction = await model.predict(tensor).data();
    const score = prediction[0];

    // Mostrar resultado
    if (score > 0.5) {
        result.textContent = `Maligno (Probabilidad: ${(score * 100).toFixed(2)}%)`;
        result.classList.add('text-red-500');
    } else {
        result.textContent = `Benigno (Probabilidad: ${((1 - score) * 100).toFixed(2)}%)`;
        result.classList.add('text-green-500');
    }

    tensor.dispose(); // Liberar memoria
});

// Cargar el modelo al iniciar
loadModel();