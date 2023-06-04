import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
 
// const path = require("path");


//TODO: FIX MODEL FILES
// ADD ABILITY TO CHOOSE SPECIFIC MODEL OUTPUT
// ADD ABILITY TO UPLOAD EXISTING PROJECT

const UploadFiles = ({projectData, setProjectData}) => {
  const [file, setFile] = useState();
  const [loading, setLoading] = useState(false) // Whether or not the model is currently running
  // const inputRef = useRef();
  // const previewRef = useRef();

  // // https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Image_types
  // const fileTypes = [
  //   "image/apng",
  //   "image/bmp",
  //   "image/gif",
  //   "image/jpeg",
  //   "image/pjpeg",
  //   "image/png",
  //   "image/svg+xml",
  //   "image/tiff",
  //   "image/webp",
  //   "image/x-icon"
  // ];

  // function validFileType(file) {
  //   return fileTypes.includes(file.type);
  // }
  
  // function updateImageDisplay() {
  //   while(previewRef.current.firstChild) {
  //     previewRef.current.removeChild(previewRef.current.firstChild);
  //   }
  
  //   const curFiles = inputRef.current.files;
  //   if (curFiles.length === 0) {
  //     const para = document.createElement('p');
  //     para.textContent = 'No files currently selected for upload';
  //     previewRef.current.appendChild(para);
  //   } else {
  //     const list = document.createElement('ol');
  //     previewRef.current.appendChild(list);
  
  //     for (const file of curFiles) {
  //       const listItem = document.createElement('li');
  //       const para = document.createElement('p');
  //       if (validFileType(file)) {
  //         para.textContent = `File name ${file.name}, file size ${returnFileSize(file.size)}.`;
  //         const image = document.createElement('img');
  //         image.src = URL.createObjectURL(file);
  
  //         listItem.appendChild(image);
  //         listItem.appendChild(para);
  //       } else {
  //         para.textContent = `File name ${file.name}: Not a valid file type. Update your selection.`;
  //         listItem.appendChild(para);
  //       }
  
  //       list.appendChild(listItem);
  //     }
  //   }
  // }

  // function returnFileSize(number) {
  //   if (number < 1024) {
  //     return `${number} bytes`;
  //   } else if (number >= 1024 && number < 1048576) {
  //     return `${(number / 1024).toFixed(1)} KB`;
  //   } else if (number >= 1048576) {
  //     return `${(number / 1048576).toFixed(1)} MB`;
  //   }
  // }

  // useEffect(() => {
  //   console.log("INPUT: ", inputRef)
  //   inputRef.current.style.opacity = 0;
  //   inputRef.current.addEventListener('change', updateImageDisplay);
  // }, [inputRef.current])

  var root_path;
  var new_root_path;
  
  function handleChange(e) {
    console.log(e.target.files);
    setFile(URL.createObjectURL(e.target.files[0]));
    const currFile = e.target.files[0]
    const currFiles = e.target.files
    // console.log("filename: ", currFile.name)
    console.log("current files:", currFiles)
    const test_file_paths = []
    for (var i = 0; i < currFiles.length; i++) {
      test_file_paths.push(currFiles[i].path)
    }
    console.log("test file path:", test_file_paths)
    console.log("current file names stringed:", JSON.stringify(test_file_paths));
    sessionStorage.setItem("fileList", JSON.stringify(test_file_paths));  

    // const new_root_path = JSON.stringify(root_path).replaceAll('\\\\', '/')
  }

  window.electronAPI.ipcR.getPath()
  .then((appDataPath) => {
      root_path = appDataPath
      new_root_path = JSON.parse(JSON.stringify(root_path).replaceAll('\\\\', '/'))
      console.log(new_root_path)
  })

  const runModel = () => {
    console.log("Hello from UploadFiles.js")
    // Do arg logic here
    const fileList = JSON.parse(sessionStorage.getItem("fileList"))
    const correctFilepaths = []
    for(var i = 0; i < fileList.length; i++) {
        const newpath = fileList[i].replaceAll("\\", "/")
        // const finalpath = "\"" + newpath + "\""
        const finalpath = newpath
        correctFilepaths.push(finalpath)
    }
    console.log(correctFilepaths)

    // const pythonArgs = ['./model_core/Model/model_checkpoint_map583.ckpt']
    // const pythonArgs = ['./resources/app/model_core/Model/model_checkpoint_map583.ckpt']
    const pythonArgs = [new_root_path + '/../model_core/Model/model_checkpoint_map583.ckpt']
    for(var i = 0; i < correctFilepaths.length; i++) {
      pythonArgs.push(correctFilepaths[i]);
    }
    // pythonArgs.push('-o ./resources/app/src/model_outputs/model_output.json')
    pythonArgs.push('-o' + new_root_path + '/../src/model_outputs/model_output.json')
    console.log("Python args: ", pythonArgs)
    window.electronAPI.ipcR.sendPythonArgs(pythonArgs);
    setLoading(true);
    window.electronAPI.ipcR.callPythonFile()
    
}
window.electronAPI.ipcR.handleScriptFinish((event, value) => {
    setLoading(false);
    window.electronAPI.ipcR.sendModelJson((event, modelJsonFile) => {
      sessionStorage.setItem("init-model", JSON.stringify(modelJsonFile));
      console.log("SENT MODEL JSON FILE FROM MAIN: ", modelJsonFile)
    })
    // var testJson = require('' + new_root_path + '/../src/model_outputs/model_output.json');
    // // var testJson = require('C:/Users/edwar/Desktop/Cal Poly/Ecology Project/forge-test-2/public/../src/model_outputs/model_output.json')
    // console.log("testJson: ", testJson)
    // setProjectData(testJson)
    // console.log("PROJECT DATA: ", projectData)
})
  
    return (
      <section className='section'>
        <h2>UploadFiles</h2>
        <Link to='/' className='btn'>
          <button>Back Home</button>
        </Link>
        <br />
        <div>
          <label htmlFor="fileUpload">Upload Images: </label>
          <br />
          {/* <input ref={inputRef} */}
          <input
            type="file" 
            id="fileUpload"
            multiple 
            onChange={handleChange} />
        </div>
        {/* <div ref={previewRef}> */}
        {/* <div>
          <p>No files currently selected for upload</p>
        </div> */}
        <br />
        <button onClick={() => runModel()}
                        className="b1"
                    >
                    Run Analysis Model 
        </button>
        <br />
        {
            !loading && 
            <h2>Model idle...</h2>
        }    
        {
            loading && 
            <h2>Running model...</h2>
        }    
        <br />
        <Link to='/modeloutput' className='btn'>
          <button>Bounding Box Editor</button>
        </Link>
      </section>
    );
  };
  export default UploadFiles;