import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './UploadFiles.css'



const UploadFiles = ({projectData, setProjectData}) => {
  const [file, setFile] = useState();
  const [loading, setLoading] = useState(false) // Whether or not the model is currently running
  const [loaded, setLoaded] = useState(false) // Whether or not the model is finished running


  var root_path;
  var new_root_path;
  
  /*
    This function takes in the list of files and saves it to session storage
    (Session storage is cleared in Home.js)
  */

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
    // console.log("test file path:", test_file_paths)
    // console.log("current file names stringed:", JSON.stringify(test_file_paths));
    sessionStorage.setItem("fileList", JSON.stringify(test_file_paths));  

    // const new_root_path = JSON.stringify(root_path).replaceAll('\\\\', '/')
  }

  /*
    This function gets the directory path from electron.js
  */

  window.electronAPI.ipcR.getPath()
  .then((appDataPath) => {
      root_path = appDataPath
      new_root_path = JSON.parse(JSON.stringify(root_path).replaceAll('\\\\', '/'))
      // console.log(new_root_path)
  })

  /*
    This function replaces the \\ with / so that the path is correct and then sends to python arguments
    to electron and calls the python file with those arguments
  */

  const runModel = () => {
    // console.log("Hello from UploadFiles.js")
    // Do arg logic here
    const fileList = JSON.parse(sessionStorage.getItem("fileList"))
    const correctFilepaths = []
    for(var i = 0; i < fileList.length; i++) {
        const newpath = fileList[i].replaceAll("\\", "/")
        // const finalpath = "\"" + newpath + "\""
        const finalpath = newpath
        correctFilepaths.push(finalpath)
    }
    // console.log(correctFilepaths)

    const pythonArgs = [new_root_path + '/../model_core/Model/model_checkpoint_map583.ckpt']
    for(var i = 0; i < correctFilepaths.length; i++) {
      pythonArgs.push(correctFilepaths[i]);
    }

    //arguments from python
    pythonArgs.push('-o' + new_root_path + '/../src/model_outputs/model_output.json')
    // console.log("Python args: ", pythonArgs)
    window.electronAPI.ipcR.sendPythonArgs(pythonArgs);
    setLoading(true);
    setLoaded(false);
    window.electronAPI.ipcR.callPythonFile()
    
}

/*
  This ipc call occurs when the python script finishes running. It saves the model output to session storage.
*/
window.electronAPI.ipcR.handleScriptFinish((event, value) => {
    setLoading(false);
    setLoaded(true);
    window.electronAPI.ipcR.sendModelJson((event, modelJsonFile) => {
      sessionStorage.setItem("init-model", JSON.stringify(modelJsonFile));
      console.log("SENT MODEL JSON FILE FROM MAIN: ", modelJsonFile)
    })
})

    return (
      <section className='section'>
        <h2>Upload Files</h2>
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
                        className="button-21"
                    >
                    Run Analysis Model 
        </button>
        <br />
        {
            (!loading && !loaded) &&
            <h2>Model idle...</h2>
        }    
        {
            (loading && !loaded) &&
            <h2>Running model...</h2>
        }    
        {
            (loaded) &&
            <h2>Model Complete... Editor Ready</h2>
        }
        <br />
        <Link to='/modeloutput' className='btn'>
          <button className='button-21'>Bounding Box Editor</button>
        </Link>
      </section>
    );
  };
  export default UploadFiles;