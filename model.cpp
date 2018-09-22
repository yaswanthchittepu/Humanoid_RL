//---------------------------------//
//  This file is part of MuJoCo    //
//  Written by Emo Todorov         //
//  Copyright (C) 2017 Roboti LLC  //
//---------------------------------//

#include <bits/stdc++.h>
#include <iostream>
#include "../include/mujoco.h"
#include "../include/glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <typeinfo>


#define SIM_TIME 0.05


using namespace std;

class Model
{

public:
	// MuJoCo data structures
	mjModel* m ;                // MuJoCo model
	mjData* d;                   // MuJoCo data
	mjData* d1;                   // MuJoCo data
	mjvCamera cam;                      // abstract camera
	mjvOption opt;                      // visualization options
	mjvScene scn;                       // abstract scene
	mjrContext con;                     // custom GPU context
	double theta[15];

	// mouse interaction
	bool button_left ;
	bool button_middle ;
	bool button_right ;
	double lastx ;
	double lasty ;
	
	//Properties
	double prec_bound;
	double precision;
	double goal[2][3];
	double state[29];
	int done[2];
	int state_dim;
	int action_dim;
	char chain;
	int end_eff;
	int stable;
	int flip;
	double distance[2];
	double action_bound[2];
	int graphics_mode;
	vector<vector<double> >  angle_bound;
	GLFWwindow* window;
	double reward[2];	
	int cf;

	Model(char set_chain='r', double set_precision=0.20, int set_grap_mode = 0)
	{	cout<<"-----------------------------------------------------------------"<<endl;
		cout<<"Model Info -- ";
		cout<<"Chain:"<<set_chain<<", Precision:"<<set_precision<<",  Graphics Mode:"<<set_grap_mode<<endl;
		cout<<"-----------------------------------------------------------------"<<endl;
		// MuJoCo data structures
		m = NULL;                  // MuJoCo model
		d = NULL;                   // MuJoCo data
		d1 = NULL;
		// mouse interaction
		button_left = false;
		button_middle = false;
		button_right =  false;
		lastx = 0;
		lasty = 0;

	    	// activate software
	 	mj_activate("/home/phani/mjpro150/bin/mjkey.txt");	
		precision = set_precision;
		chain  = set_chain;
		stable = 1;
		action_bound[0]=-0.08, action_bound[1]=0.08;
		action_dim = 13;
		state_dim = 28;
		flip = 0;
		graphics_mode = set_grap_mode;
		angle_bound.clear();
		vector<double> tmp_vec = {-1.070719,-1.460350,-0.917321,-0.380427,-2.144505,-0.386563,-0.523599,-0.610865,-1.221730,-0.523599,-0.349066,-3.126253,-0.017453,-1.396263,-1.919862,-3.126253,-3.141593,-1.396263,-1.919862,-0.386563,-2.144505,-1.920544,-0.003068,-0.380427,-1.070719 };
		angle_bound.push_back(tmp_vec);
		vector<double> tmp_vec1 = {1.230253,0.380427,0.003068,1.920544,1.920544,0.380427,0.785398,0.610865,1.221730,0.785398,0.349066,3.071030,3.141593,1.396263,-0.017453,3.071030,0.017453,1.396263,-0.017453,0.380427,1.920544,0.380427,0.917321,1.460350,1.230253}; 


		angle_bound.push_back(tmp_vec1);
		//angle_bound = new int[][]{{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1  } , {1,1,1,1,1,1,1,1,1,1,1,1,1}};

		cf = 0;

		if(chain == 'r')
			end_eff = 0;
		else if(chain=='l')
			end_eff = 1;
		else
			cout<<"Error in defining chain"<<endl;
	



	// main function

	
	    // check command-line arguments
	    // if( argc!=2 )
	    // {
	    //     printf(" USAGE:  basic modelfile\n");
	    //     return 0;
	    // }

	 

	    // load and compile model
	    char error[1000] = "Could not load binary model";
	    m = mj_loadXML("/home/phani/mjpro150/model/new_model.xml", 0, error, 1000);
	    // char error[1000] = "Could not load binary model";
	    // if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
	    //     m = mj_loadModel(argv[1], 0);
	    // else
	    //     m = mj_loadXML(argv[1], 0, error, 1000);
	    // if( !m )
	    //     mju_error_s("Load model error: %s", error);

	    // make data
	    d = mj_makeData(m);
	    d1 = mj_makeData(m);
	    srand (time(NULL));
	    cout<<"-------------Model Created------------"<<endl;
	    if(graphics_mode){
		    // init GLFW
		    if( !glfwInit() )
			    mju_error("Could not initialize GLFW");

		    // create window, make OpenGL context current, request v-sync
		        // GLFWwindow* window = glfwCreateWindow(1200, 900, "Simulate", NULL, NULL);

		    window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
		    glfwMakeContextCurrent(window);
		    glfwSwapInterval(1);

		    // initialize visualization data structures
		    mjv_defaultCamera(&cam);
		    mjv_defaultOption(&opt);
		    mjr_defaultContext(&con);
		    mjv_makeScene(&scn, 1000);                   // space for 1000 objects
		    mjr_makeContext(m, &con, mjFONTSCALE_100);   // model-specific context

		    // install GLFW mouse and keyboard callbacks
		    // make the member variables static in order to use the below functions, like static mft m;
		    //glfwSetKeyCallback(window, Model::keyboard);// std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4, std::placeholders::_5));
		    //glfwSetCursorPosCallback(window, mouse_move);
		    //glfwSetMouseButtonCallback(window, mouse_button);
		    //glfwSetScrollCallback(window, scroll);

		    // std::cout << m->nu<<std::endl;
		    // run main loop, target real-time simulation and 60 fps rendering
	    }
	    stable = 1;
	    done[0] = 0;
	    done[1] = 0;
	    prec_bound = 0.02;

	}
	

	void table_sample()
	{

		/*double height_z[2] = {0.525, 0.72};
		double yx_start[2] = {-0.25,-0.275};
		double yx_end[2] = {0.1, -0.125};
	
		goal[0][2] = goal[1][2] = height_z[flip];
		double obj_center_y = fRand(yx_start[0], yx_end[0]);	
		double obj_center_x = fRand(yx_start[1], yx_end[1]);
		
		m->body_pos[28*3-3] = obj_center_x;
		m->body_pos[28*3-2] = obj_center_y;
		m->body_pos[28*3-1] = height_z[flip];
		//cout << height_z[flip] << endl;

		goal[0][0] = goal[1][0] = obj_center_x;
		goal[1][1] = obj_center_y - 0.025; // left 	
		goal[0][1] = obj_center_y + 0.025; //right*/

		double base_joint_range[2] = {-1.57079632679, 1.57079632679}
		double chest_side_joint_range[2] = {-1.16937059884, 0.471238898038}
		double chest_front_joint_range[2] = {-0.698131700798, 0.698131700798}
		double left_shoulder_joint_range[2] = {-2.09439510239, 2.70526034059}
		double left_shoulder2_joint_range[2] = {-1.83259571459, 1.91986217719}
		double left_elbow_joint_range[2] = {-1.83259571459, 1.83259571459}
		double left_elbow2_joint_range[2] = {-2.58308729295, 0.0174532925199}
		double left_wrist_joint_range[2] = {0.00, 1.57079632679}
		double neck_joint_range[2] = {-1.57079632679, 1.57079632679}
		double head_joint_range[2] = {-0.785398163397, 0.10471975512}
		double right_shoulder_joint_range[2] = {-2.70526034059, 2.09439510239}
		double right_shoulder2_joint_range[2] = {-1.91986217719, 1.83259571459}
		double right_elbow_joint_range[2] = {-1.83259571459, 1.83259571459}
		double right_elbow2_joint_range[2] = {-0.0174532925199, 2.58308729295}
		double right_wrist_joint_range[2] = {-1.57079632679, 0.00}

	}

	void sphere_sample()
	{
		double hlimit_z[2]={0.5, 0.70};
		double YX_start[2] = {-0.25,-0.275};
		double YX_end[2] = {0.1, -0.105};
                double sph_center_y= fRand(YX_start[0], YX_end[0]);//-0.01*goal[1][1];
                              
		double sph_center_x = fRand(YX_start[1], YX_end[1]);
                while(sph_center_y == goal[1][1] && sph_center_x == goal[1][0])
                   {
                     sph_center_y = fRand(YX_start[0], YX_end[0]);
                     sph_center_x = fRand(YX_start[1], YX_end[1]);
                   }
                   
		double sph_height_z = fRand(hlimit_z[0], hlimit_z[1]);
		
		m->body_pos[29*3-3]=sph_center_x;
		m->body_pos[29*3-2]=sph_center_y;
		m->body_pos[29*3-1]=sph_height_z;

	}


	void Reset()
	{	
	    //flip = !(flip);

	    cf = 0;
	    
	    // Make all the angles zero
	    mj_resetData(m,d);
    
	    //cout << m->body_pos[28*3 - 3] <<" "<< m->body_pos[28*3 - 2] << " " << m->body_pos[28*3 - 1]<<endl;
	    
	
            //sphere_sample();
	
	    //Set Box goals
	    table_sample();   

	    //Start Simulation
	    mj_step(m,d);
   
	     
	    double act[13] = {0};
	    
	    /*if(d->site_xpos[8]>0.008)
			stable = 0;
	    else
			stable = 1;
		*/
	    done[0] = 0;
	    done[1] = 0;
	    Get_state();


	}
	

	void Get_state()
	{
		for(int i = 0; i < 8; i++)
		{
			state[i] = d->qpos[i]
		}
		for(int i = 8; i < 10; i++)
		{
			state[i] = d->qpos[5+i]
		}
		for(int i = 10; i < 15; i++)
		{
			state[i] = d->qpos[i-2]
		}
		for(int i=0;i<3;i++)
		{
			state[15+i] = goal[0][i];
		}
		for(int i=0;i<3;i++)
		{
			state[18+i] = goal[1][i];
		}
		for(int i=0;i<3;i++)
		{
			state[21+i] = d->site_xpos[0*3+i];
		}
		for(int i=0;i<3;i++)
		{
			state[24+i] = d->site_xpos[1*3+i];
		}
		state[27] = done[0];
		state[28] = done[1];
		state[29] = cf;
	}
		/*for(int i=0;i<13;i++)
		{
			state[i] = d->qpos[6+i];
		}
		for(int i=0;i<3;i++)
		{
			state[13+i] = goal[0][i];
		}
		for(int i=0;i<3;i++)
		{
			state[16+i] = goal[1][i];
		}
		for(int i=0;i<3;i++)
		{
			state[19+i] = d->site_xpos[0*3+i];
		}
		for(int i=0;i<3;i++)
		{
			state[22+i] = d->site_xpos[1*3+i];
		}*/

		/*
		int j = 0;
		for(int i=0;i<42;i=i+6)
		{	
			state[25+i] = m->geom_pos[(35+j)*3];
			state[25+i+1] = m->geom_pos[(35+j)*3 + 1];
			state[25+i+2] = m->geom_pos[(35+j)*3 + 2];
			state[25+i+3] = m->geom_size[(35+j)*3 + 1];
			state[25+i+4] = m->geom_size[(35+j)*3 + 2];
			state[25+i+5] = m->geom_size[(35+j)*3 + 3];
 			j++;
		}
		//cout << m->site_size[3*3] << endl;
		//cout << m->geom_pos[37*3-3] << m->geom_pos[37*3-2] << m->geom_pos[37*3-1] << endl;
		//cout << m->geom_pos[38*3-3] << m->geom_pos[38*3-2] << m->geom_pos[38*3-1] << endl;
		*/
		//state[25] = done[0];
		//state[26] = done[1];
		//state[27] = cf;
	}

	void Reward()
	{

		done[0] = 0;	
		done[1] = 0;	
		cf = 0;
		double sum = 0.0;
		double sum1 = 0.0;
		reward[0] = reward[1] = 0.0;
		
		for(int i=0;i<3;i++){
			sum+= pow(d->site_xpos[0*3+i] - goal[0][i],2);
			sum1+= pow(d->site_xpos[1*3+i] - goal[1][i],2);
		}
		double dist = sqrt(sum);
		double dist1 = sqrt(sum1);
		

		reward[0] = -10*dist;
		reward[1] = -10*dist1;

		// If robot falls
		// cout<<d->site_xpos[6]<<' '<<d->site_xpos[7]<<' '<<d->site_xpos[8]<<endl;
		
		// site2 - left leg
		if(d->site_xpos[8]>0.008){
			stable = 0;
			reward[0] = -30;
			reward[1] = -30;
			return ;

		}

		if(dist<prec_bound){

			reward[0]+= 5;
			if(dist < precision){
				reward[0]+=10;
				done[0] = 1;
			}
		}
		
		if(dist1<prec_bound){

			reward[1]+= 5;
			if(dist1 < precision){
				reward[1]+=10;
				done[1] = 1;
			}
		}



		if(dist<precision && dist1<precision)
		{
			reward[0]+=20;
			reward[1]+=20;
			done[0]=1;
			done[1]=1;
		}
			int cc = check_collision();
		if(cc){
			reward[0]=-15;
			reward[1]=-15;
			cf = 1;
			//cout << d->sensordata[0] << endl; 
		}
		
		distance[0] = dist;
		distance[1] = dist1;
		
	
	}

	int check_collision()
	{	
		for(int i=0;i<7;i++)
		{	
			
			if(d->sensordata[i]>0)
				return 1;

		}
	return 0;
	}

	double clip(double act, int idx)
	{
		act = max(act, angle_bound[0][idx]);
		act = min(act, angle_bound[1][idx]);
		return act;


	}


	double* Step(double action[13])
	{


	    //for(int i=0;i<13;i++)
		//cout<<action[i]<<' ';
	    //while( !glfwWindowShouldClose(window) )
	    //{
	        // advance interactive simulation for 1/60 sec
	        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
	        //  this loop will finish on time for the next frame to be rendered at 60 fps.
	        //  Otherwise add a cpu timer and exit this loop when it is time to render.
	        mjtNum simstart = d->time;
	        for (int i=0;i<13;i++){
	            d->ctrl[6+i]+=action[i];
	            d->ctrl[6+i] = clip(d->ctrl[6+i], 6+i);
	        }
	        
		
	        while( d->time - simstart < SIM_TIME ){
	            

	           mj_step(m, d);
	           //mj_kinematics(m, d);

	            //for(int i=0;i<m->nq;i++){
	                // std::cout << d->xpos[29*3+i] <<std::endl;
	                // std::cout << d->actuator_length[i] <<std::endl;

	              //  std::cout <<i<<' '<< d->qpos[i] <<std::endl;

	                // std::cout << d->subtree_com[i] <<std::endl;
	                // std::cout << m->body_subtreemass[i] <<std::endl;
	           // }
	           // printf("-------\n");
	        
	        
		    // std::cout<<d->qpos[50]<<std::endl;
	        // get framebuffer viewport
		
		}
		//for (int j=0; j<m->nsensordata;j++)
		//	cout<<d->sensordata[j]<<' ';
		//cout<<endl;
		
		if(graphics_mode)
		{
			mjrRect viewport = {0, 0, 0, 0};
			glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
			mjrRect smallrect = viewport;

			// update scene and render
			mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
			mjr_render(viewport, &scn, &con);

			// swap OpenGL buffers (blocking call due to v-sync)
			glfwSwapBuffers(window);

			// process pending GUI events, call GLFW callbacks
			glfwPollEvents();

		}

		Reward();
		Get_state();
		return reward;
	    
	}


	void Close()
	{
	    

	        // close GLFW, free visualization storage
		if(graphics_mode)
		{
			glfwTerminate();
			mjv_freeScene(&scn);
			mjr_freeContext(&con);
		}
		// free MuJoCo model and data, deactivate
	   	mj_deleteData(d);
	    	mj_deleteModel(m);
	    	mj_deactivate();

	    // return 1;
	}

	
	double fRand(double fMin, double fMax)
	{
    		double f = (double)rand() / RAND_MAX;
    		return fMin + f * (fMax - fMin);
	}
	

	// keyboard callback
	void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
	{
	    // backspace: reset simulation
	    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
	    {
	        mj_resetData(m, d);
	        mj_forward(m, d);
	    }
	}


	// mouse button callback
	void mouse_button(GLFWwindow* window, int button, int act, int mods)
	{
	    // update button state
	    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
	    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
	    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

	    // update mouse position
	    glfwGetCursorPos(window, &lastx, &lasty);
	}


	// mouse move callback
	void mouse_move(GLFWwindow* window, double xpos, double ypos)
	{
	    // no buttons down: nothing to do
	    if( !button_left && !button_middle && !button_right )
	        return;

	    // compute mouse displacement, save
	    double dx = xpos - lastx;
	    double dy = ypos - lasty;
	    lastx = xpos;
	    lasty = ypos;

	    // get current window size
	    int width, height;
	    glfwGetWindowSize(window, &width, &height);

	    // get shift key state
	    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
	                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

	    // determine action based on mouse button
	    mjtMouse action;
	    if( button_right )
	        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
	    else if( button_left )
	        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
	    else
	        action = mjMOUSE_ZOOM;

	    // move camera
	    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
	}


	// scroll callback
	void scroll(GLFWwindow* window, double xoffset, double yoffset)
	{
	    // emulate vertical mouse motion = 5% of window height
	    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
	}


};



extern "C"
{
    Model* Model_new(char set_chain, double set_precision, int set_grap_mode) {return new Model(set_chain,set_precision, set_grap_mode);}
    void Model_reset(Model* model) {model->Reset();}
    double * Model_get_state(Model* model) {return model->state;}
    double * Model_get_distance(Model* model) {return model->distance;}
    int Model_get_stable(Model* model) {return model->stable;}
    int Model_get_state_dim(Model* model) {return model->state_dim;}
    int Model_get_action_dim(Model* model) {return model->action_dim;}
    double * Model_get_action_bound(Model* model) {return model->action_bound;}
    void Model_close(Model* model) {model->Close();}
    double * Model_step(Model* model, double action[13]) {return model->Step(action);}
    
    // void Foo_foobar(Foo* foo, int n) {return foo->foobar(n);}
}

// ------------------------------






int main()
{
//Model* m = new Model();
//m->Reset();
return 0;
}

//     #include <iostream>
//     // A simple class with a constuctor and some methods...
//     class Foo
//     {
//         public:
//             Foo(int);
//             void bar();
//             int foobar(int);
//         private:
//             int val;
//     };
//     Foo::Foo(int n)
//     {
//         val = n;
//     }
//     void Foo::bar()
//     {
//         std::cout << "Value is " << val << std::endl;
//     }
//     int Foo::foobar(int n)
//     {
//         return val + n;
//     }

//     extern "C"
// {
//     Foo* Foo_new(int n) {return new Foo(n);}
//     void Foo_bar(Foo* foo) {foo->bar();}
//     int Foo_foobar(Foo* foo, int n) {return foo->foobar(n);}
// }


