import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';


const ExistingProject = () => {

    const [uploaded, setUploaded] = useState(false) // Whether or not a file is uploaded

    /*
        This function prompts the promtps the user to choose an existing project, and saves the json data and generated fileList to sessionStorage
    */
    function handleChange() {
        window.electronAPI.ipcR.openProjectData();
        
        window.electronAPI.ipcR.sendModelJson((event, modelJsonFile) => {
            var existing_file_paths = []
            // Generate fileList like in UploadFiles.js
            Object.keys(modelJsonFile).forEach(function(imgPath) {
                existing_file_paths.push(imgPath.replaceAll('/', '\\'));
            });

            console.log("Existing paths stringed: ", JSON.stringify(existing_file_paths))
            sessionStorage.setItem("fileList", JSON.stringify(existing_file_paths));  
            sessionStorage.setItem("init-model", JSON.stringify(modelJsonFile));
            console.log("SENT MODEL JSON FILE FROM MAIN: ", modelJsonFile);
            setUploaded(true);

        })
    }



    return (
        <section className='section'>
            <h2>ExistingProject</h2>
            <Link to='/' className='btn'>
                <button>Back Home</button>
            </Link>

            <div>
                <label htmlFor="projUpload">Upload Existing Project (json): </label>
                <br />
                <button onClick={() => handleChange()}
                        className="b1"
                >
                    Choose Project File
                </button>
                <br />
                {
                    uploaded && 
                    <h2>Project Uploaded</h2>
                }
            </div>
            <Link to='/modeloutput' className='btn'>
                <button>Bounding Box Editor</button>
            </Link>
        </section>
      );
    };
    export default ExistingProject;