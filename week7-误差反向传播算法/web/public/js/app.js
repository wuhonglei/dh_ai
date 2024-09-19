var wrapper = document.getElementById("signature-pad");
var clearButton = wrapper.querySelector("[data-action=clear]");
var changeColorButton = wrapper.querySelector("[data-action=change-color]");
var undoButton = wrapper.querySelector("[data-action=undo]");
var savePNGButton = wrapper.querySelector("[data-action=save-png]");
var saveJPGButton = wrapper.querySelector("[data-action=save-jpg]");
var saveSVGButton = wrapper.querySelector("[data-action=save-svg]");
const identityButton = document.querySelector("[data-action=identify]");
var canvas = wrapper.querySelector("canvas");

var signaturePad = new SignaturePad(canvas, {
  // It's Necessary to use an opaque color when saving image as JPEG;
  // this option can be omitted if only saving as PNG or SVG
  backgroundColor: "rgb(0, 0, 0)",
  penColor: "rgb(255, 255, 255)",
  minWidth: 20, // 设置线条的最小宽度
  maxWidth: 24, // 设置线条的最大宽度
});

// Adjust canvas coordinate space taking into account pixel ratio,
// to make it look crisp on mobile devices.
// This also causes canvas to be cleared.
function resizeCanvas() {
  // When zoomed out to less than 100%, for some very strange reason,
  // some browsers report devicePixelRatio as less than 1
  // and only part of the canvas is cleared then.
  var ratio = Math.max(window.devicePixelRatio || 1, 1);

  // This part causes the canvas to be cleared
  canvas.width = canvas.offsetWidth * ratio;
  canvas.height = canvas.offsetHeight * ratio;
  canvas.getContext("2d").scale(ratio, ratio);

  // This library does not listen for canvas changes, so after the canvas is automatically
  // cleared by the browser, SignaturePad#isEmpty might still return false, even though the
  // canvas looks empty, because the internal data of this library wasn't cleared. To make sure
  // that the state of this library is consistent with visual state of the canvas, you
  // have to clear it manually.
  signaturePad.clear();
}

// On mobile devices it might make more sense to listen to orientation change,
// rather than window resize events.
// window.onresize = resizeCanvas;
resizeCanvas();

function download(dataURL, filename) {
  var blob = dataURLToBlob(dataURL);
  var url = window.URL.createObjectURL(blob);

  var a = document.createElement("a");
  a.style = "display: none";
  a.href = url;
  a.download = filename;

  document.body.appendChild(a);
  a.click();

  window.URL.revokeObjectURL(url);
}

// One could simply use Canvas#toBlob method instead, but it's just to show
// that it can be done using result of SignaturePad#toDataURL.
function dataURLToBlob(dataURL) {
  // Code taken from https://github.com/ebidel/filer.js
  var parts = dataURL.split(";base64,");
  var contentType = parts[0].split(":")[1];
  var raw = window.atob(parts[1]);
  var rawLength = raw.length;
  var uInt8Array = new Uint8Array(rawLength);

  for (var i = 0; i < rawLength; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }

  return new Blob([uInt8Array], { type: contentType });
}

clearButton?.addEventListener("click", function (event) {
  signaturePad.clear();
});

undoButton?.addEventListener("click", function (event) {
  var data = signaturePad.toData();

  if (data) {
    data.pop(); // remove the last dot or line
    signaturePad.fromData(data);
  }
});

changeColorButton?.addEventListener("click", function (event) {
  var r = Math.round(Math.random() * 255);
  var g = Math.round(Math.random() * 255);
  var b = Math.round(Math.random() * 255);
  var color = "rgb(" + r + "," + g + "," + b + ")";

  signaturePad.penColor = color;
});

savePNGButton?.addEventListener("click", function (event) {
  if (signaturePad.isEmpty()) {
    alert("Please provide a signature first.");
  } else {
    var dataURL = signaturePad.toDataURL();
    download(dataURL, "signature.png");
  }
});

saveJPGButton?.addEventListener("click", function (event) {
  if (signaturePad.isEmpty()) {
    alert("Please provide a signature first.");
  } else {
    const input = document.querySelector(".name");
    const name = input.value || "signature.jpg";
    const pattern = /\d_(\d+)/;
    const match = name.match(pattern);
    var dataURL = signaturePad.toDataURL("image/jpeg");
    input.value = match
      ? match[0].replace(`_${match[1]}`, `_${Number(match[1]) + 1}`)
      : "";
    download(dataURL, name);
    signaturePad.clear();
  }
});

saveSVGButton?.addEventListener("click", function (event) {
  if (signaturePad.isEmpty()) {
    alert("Please provide a signature first.");
  } else {
    var dataURL = signaturePad.toDataURL("image/svg+xml");
    download(dataURL, "signature.svg");
  }
});

async function predictFetchList(dataURL, label, models) {
  const promise = models.map((modelName) =>
    fetch("/predict", {
      method: "POST", // 使用 POST 方法
      headers: {
        "Content-Type": "application/json", // 指定请求体的内容类型为 JSON
      },
      body: JSON.stringify({
        label,
        dataURL,
        modelName: modelName,
      }),
    }).then((res) => res.json())
  );

  const resList = await Promise.all(promise);
  return resList; // [{ prediction, probability, probabilities }]
}

identityButton?.addEventListener("click", async function (event) {
  if (signaturePad.isEmpty()) {
    alert("Please provide a signature first.");
  } else {
    const input = document.querySelector(".name");
    const model1 = document.querySelector("#model1");
    const model2 = document.querySelector("#model2");

    const name = input.value || "signature.jpg";
    const pattern = /(\d)(_\d+)?/;
    const match = name.match(pattern);
    const label = match ? Number(match?.[1]) : undefined;

    var dataURL = signaturePad.toDataURL("image/jpeg");
    const resList = await predictFetchList(dataURL, label, [
      model1.value,
      model2.value,
    ]);
    [".predict-container-left", ".predict-container-right"].forEach(
      (selector, index) => {
        const container = document.querySelector(selector);
        const predictDigit = container.querySelector(".predict-digit");
        const predictProb = container.querySelector(".predict-prob");
        const predictList = container.querySelector(".predict-list");

        const { prediction, probability, probabilities } = resList[index];
        predictDigit.textContent = prediction;
        predictProb.textContent = `${(probability * 100).toFixed(2)}%`;
        predictList.innerHTML = probabilities
          .map((prob, index) => {
            const li = document.createElement("li");
            li.textContent = `${index}: ${(prob * 100).toFixed(2)}%`;
            if (index === prediction) {
              li.classList.add("predict-index");
            }
            if (label == index) {
              li.classList.add("predict-success");
            }

            return li.outerHTML;
          })
          .join("");
      }
    );
  }
});

document.addEventListener("keydown", function (event) {
  if ((event.metaKey || event.ctrlKey) && event.key === "z") {
    // 你的处理逻辑
    undoButton?.click();
  }
});
