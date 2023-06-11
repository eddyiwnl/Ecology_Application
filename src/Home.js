import { Link } from 'react-router-dom';
import './Home.css'

const Home = () => {
  sessionStorage.clear();
  
  return (
    <section className='section'>
    <h2>Sample Analyzer</h2>
      <Link to='/uploadfiles' className='btn'>
        <button className='button-21'>New Project</button>
      </Link>
      <br></br> 
      <Link to='/existingproject' className='btn'>
        <button className='button-21'>Open an Existing Project</button>
      </Link>
      <br></br>
      {/* <Link to='/modeltest' className='btn'>
        Model Testing
      </Link>  */}
      {/* <br />
      <Link to='/modeloutput' className='btn'>
          Bounding Box Canvas
      </Link> */}
    </section>
  );
};
export default Home;

