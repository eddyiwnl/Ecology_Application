import { Link } from 'react-router-dom';


const ExistingProject = () => {

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
        })

        


        // console.log(e.target.files);
        // // setFile(URL.createObjectURL(e.target.files[0]));
        // const currFile = e.target.files[0]
        // const currFiles = e.target.files
        // // console.log("filename: ", currFile.name)
        // console.log("current files:", currFiles)
        // console.log("test: ", currFiles[0].path)
        // const proj_file_path_list = currFiles[0].path.replaceAll("\\", "/").split("/")
        // const proj_file_path = proj_file_path_list[proj_file_path_list.length - 1]
        
        // console.log("Project file path:", proj_file_path)

        // var proj_file = require('./resources/app/src/model_outputs/' + proj_file_path)
        // // var proj_file = require('./model_outputs/' + 'model_outputs.json')
        // // var proj_file = require('./model_outputs/test_save1.json')



        // // console.log(proj_file)
        // sessionStorage.setItem("curr_json", JSON.stringify(proj_file))
        // // sessionStorage.setItem("fileList", JSON.stringify(test_file_paths));  

        // const testing = JSON.parse(sessionStorage.getItem("curr_json"));
        // console.log(testing)
        // for (var key in testing) {
        //     if (testing.hasOwnProperty(key)) {
        //         console.log(key + " -> " + testing[key]);
        //     }
        // }
        // // for(var i=0; i<Object.keys(testing).length; i++) {
        // //     console.log(testing[i])
        // // }
    }



    return (
        <section className='section'>
            <h2>ExistingProject</h2>
            <Link to='/' className='btn'>
                Back Home
            </Link>

            <div>
                <label htmlFor="projUpload">Upload Existing Project (json): </label>
                <br />
                {/* <input ref={inputRef} */}
                {/* <input
                    type="file" 
                    id="projUpload"
                    onChange={handleChange} /> */}
                <button onClick={() => handleChange()}
                        className="b1"
                >
                    Choose Project File
                </button>
            </div>
            <Link to='/modeloutput' className='btn'>
                Bounding Box Editor
            </Link>
        </section>
      );
    };
    export default ExistingProject;