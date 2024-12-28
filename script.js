const imageUpload = document.getElementById("imageUpload");
const container = document.getElementById("container");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(start);

async function start() {
  const labeledFaceDescriptors = await loadLabeledImages();
  document.body.append("Loaded");
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

  imageUpload.addEventListener("change", async () => {
    container.innerHTML = "";
    // FaceApi loads up the image
    const image = await faceapi.bufferToImage(imageUpload.files[0]);
    // Append the mage to the div container
    container.append(image);
    // Create a canvas element to display the boxes
    const canvas = faceapi.createCanvasFromMedia(image);
    // append it to the container
    container.append(canvas);
    // Create the size object
    const displaySize = {
      width: image.width,
      height: image.height,
    };
    faceapi.matchDimensions(canvas, displaySize);
    // Detect the faces
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = await faceapi.resizeResults(
      detections,
      displaySize
    );
    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );

    // Draw box on each faces
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
      });
      drawBox.draw(canvas);
    });
  });
}

async function loadLabeledImages() {
  const labels = [
    "Black Widow",
    "Captain America",
    "Captain Marvel",
    "Hawkeye",
    "Jim Rhodes",
    // "Nebula",
    "Thor",
    "Tony Stark",
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (let i = 1; i < 2; i++) {
        const img = await faceapi.fetchImage(
          `https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        descriptions.push(detections.descriptor);
      }
      //   console.log(descriptions);
      // console.log(faceapi.LabeledFaceDescriptors(label, descriptions));

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}
