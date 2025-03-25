/*
 * Copyright (c) 2025 Vito Romanelli Tricanico
 *
 * SPDX-License-Identifier: BSD-2-Clause (see License)
 */
 
// usage: ./raytracer <resolution>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define OFFSET 1e-5
#define n_AIR 1.0003

typedef struct color{
	double R, G, B;
} color;

typedef struct material{
	color m_a, m_d, m_s; // ambience, diffusiveness and specularity
	double m_sm, m_sp; // metalness and phong exponent
	double n[2]; // refractive index (complex: if transparent, n[1] = 0.0 usually)
} material;

typedef struct sphere{
	double xc, yc, zc; // positions of center of sphere
	double r; // radius
	material M;
	color c;
	int ID;
} sphere;

typedef struct ray{
	double ox, oy, oz; // position of origin of ray
	double dx, dy, dz; // unit vector values (direction)
} ray;

void init_color(color *C, double R, double G, double B){
	C->R = R;
	C->G = G;
	C->B = B;
}

void init_material(material *M, color m_a, color m_d, color m_s, double m_sm, double m_sp, double nr, double ni){
	M->m_a = m_a;
	M->m_d = m_d;
	M->m_s = m_s;
	M->m_sm = m_sm;
	M->m_sp = m_sp;
	M->n[0] = nr;
	M->n[1] = ni;
}

void init_sphere(sphere *s, double x, double y, double z, double r, int ID, color C, material M){
    if (s != NULL) {
        s->xc = x;
        s->yc = y;
        s->zc = z;
        s->r = r;
        s->ID = ID;
        s->c = C;
        s->M = M;
    }
}

int sign(double num){
	if (num >= 0.0) return 1;
	else if (num < 0.0) return -1;
}

// computes the dot product between 2 arrays
double dot(double u[], double v[]){
	return u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
}

// normalize vector
void norm_vector(double *v){
    double abs_s = sqrt(dot(v, v));
    if (abs_s != 0) { // Check to avoid division by zero
        v[0] /= abs_s;
        v[1] /= abs_s;
        v[2] /= abs_s;
    }
}

// Maybe needs adjustment for aspect ratio.
// if the screen plane is perpendicular to the z axis, only x and y will vary for the pixel position
// so pix_z is screen[2]
void compute_prime(ray *primeray, double eye[], double pix_x, double pix_y, double pix_z){
	double u[3];
	
	primeray->ox = eye[0];
	primeray->oy = eye[1];
	primeray->oz = eye[2];
	
	u[0] = pix_x - eye[0];
	u[1] = pix_y - eye[1];
	u[2] = pix_z - eye[2];
	
	norm_vector(u);
	
	primeray->dx = u[0];
	primeray->dy = u[1];
	primeray->dz = u[2];
	
}

// routine that computes values on p_hit array, N_hit and the return the intersect boolean
bool intersect(sphere *s, ray *r, double *p_hit, double *N_hit, bool outside){
	double a, b, c, q, t1, t2, t, dis;
	bool retval; // return val
	a = r->dx*r->dx + r->dy*r->dy + r->dz*r->dz;
	b = 2*((r->ox - s->xc)*r->dx + (r->oy - s->yc)*r->dy + (r->oz - s->zc)*r->dz);
	c = (r->ox - s->xc)*(r->ox - s->xc) + (r->oy - s->yc)*(r->oy - s->yc) + (r->oz - s->zc)*(r->oz - s->zc) - s->r*s->r;
	dis = b*b - 4*a*c; // discriminant
	if (dis >= 0){
	
		q = -0.5*(b + (double)sign(b)*sqrt(dis));
		t1 = q/a;
		t2 = c/q;
		t = fmin(t1, t2);
		
		if (t <= 0){
			retval = false;
		}
		else {
			retval = true;
			
			// hit coordinates (not a direction vector)
			p_hit[0] = r->dx*t + r->ox;
			p_hit[1] = r->dy*t + r->oy;
			p_hit[2] = r->dz*t + r->oz;
			
			N_hit[0] = (p_hit[0] - s->xc)/s->r;
			N_hit[1] = (p_hit[1] - s->yc)/s->r;
			N_hit[2] = (p_hit[2] - s->zc)/s->r;
			/*
			if (outside == false){
				N_hit[0] = -N_hit[0];
				N_hit[1] = -N_hit[1];
				N_hit[2] = -N_hit[2];
			}
			*/
			//printf("%lf\n", sqrt(dot(N_hit, N_hit))); // check that N is already normalized
		}
	}
	else {
		retval = false;
	}
	return retval;
}

// computes distance between 2 vectors
// can be used to compute magnitude if v = {0, 0 ,0}
double distance(double u[], double v[]){
	double xx, yy, zz;
	xx = u[0] - v[0];
	yy = u[1] - v[1];
	zz = u[2] - v[2];
	return sqrt(xx*xx + yy*yy + zz*zz);
}

void compute_reflected(double *ref, double *inc, double *N_hit){

	ref[0] = inc[0] - 2*(dot(inc, N_hit))*N_hit[0];
	ref[1] = inc[1] - 2*(dot(inc, N_hit))*N_hit[1];
	ref[2] = inc[2] - 2*(dot(inc, N_hit))*N_hit[2];
	
	norm_vector(ref);
}

// Compute final RGB colors with phong shading (for single light source) for opaque objects:
//
// C is the RGB array color to be computed on pixels[i][j]
// A is the RGB array for ambient light
// Im is the RGB intensity of the infalling light
// Lm is the ray from the object towards the light source (could be a shadow ray)
// N is the normal array on given point
// V is the array from the camera towards the object, inverted
color rho(sphere *s, double *A, double *Im, ray *Lm, double *N, ray *V){
	double amb[3], dif[3], spc[3], S[3], Rm[3], Lm_N, Rm_V, W1, W2, auxL[3], auxV[3];
	color C;

	auxL[0] = Lm->dx;
	auxL[1] = Lm->dy;
	auxL[2] = Lm->dz;
	
	auxV[0] = V->dx;
	auxV[1] = V->dy;
	auxV[2] = V->dz;
	
	compute_reflected(Rm, auxL, N);
	
	Lm_N = dot(auxL, N);
	Rm_V = dot(Rm, auxV);
	
	S[0] = s->M.m_sm * s->c.R + (1 - s->M.m_sm);
	S[1] = s->M.m_sm * s->c.G + (1 - s->M.m_sm);
	S[2] = s->M.m_sm * s->c.B + (1 - s->M.m_sm);
	
	amb[0] = s->M.m_a.R * s->c.R * A[0];
	amb[1] = s->M.m_a.G * s->c.G * A[1];
	amb[2] = s->M.m_a.B * s->c.B * A[2];
	
	W1 = fmax(Lm_N, 0.0);
	W2 = pow(fmax(Rm_V, 0.0), s->M.m_sp); // pow is expensive
	
	// W2_optimized = (max(1-lambda)^beta, 0)^gamma, where lambda = 1 - dot(Rm, V) and beta = m_sp/gamma
	// = max(1 - beta*gamma, 0)^gamma
	// Choose gamma as gamma = 2^n, where n is integer, like 4 (just 3 multiplies) or 8 (3 multiplies), then
	// lambda = dot((Rm - V),(Rm - V))/2.
	
	//printf("%lf %lf\n", W1, W2);
	
	dif[0] = s->M.m_d.R * s->c.R * Im[0] * W1;
	dif[1] = s->M.m_d.G * s->c.G * Im[1] * W1;
	dif[2] = s->M.m_d.B * s->c.B * Im[2] * W1;
	
	spc[0] = s->M.m_s.R * S[0] * Im[0] * W2;
	spc[1] = s->M.m_s.G * S[1] * Im[1] * W2;
	spc[2] = s->M.m_s.B * S[2] * Im[2] * W2;
	
	//printf("%lf %lf %lf\n", amb[0], dif[0], spc[0]);
	
	C.R = amb[0] + dif[0] + spc[0];
	C.G = amb[1] + dif[1] + spc[1];
	C.B = amb[2] + dif[2] + spc[2];
	
	return C;
}

// anti-aliasing filter, based on contrast sampling
void aaf(color **pixels, int PIX_x, int PIX_y, int max_steps){
	int steps = 0;
	while(steps < max_steps){
		#pragma omp parallel for
		for (int i = 1; i < PIX_x - 1; i++){
			for (int j = 1; j < PIX_y - 1; j++){
				
					// to compute correct weighting:
					
					// compute the contrast between pixels[i][j].R and pixels[i+1][j].R as
					// c0 = fabs(pixels[i][j].R - pixels[i+1][j].R),
					// then rescale it with c0 = (1.0 - c0). Then the weight w is
					// 
					// w = c0/sum(c0 + c1 + c2 + c3)
					
					double wR0, wR1, wR2, wR3, wR;
					double wG0, wG1, wG2, wG3, wG;
					double wB0, wB1, wB2, wB3, wB;
					
					wR0 = 1.0 - fabs(pixels[i+1][j].R - pixels[i][j].R);
					wR1 = 1.0 - fabs(pixels[i-1][j].R - pixels[i][j].R);
					wR2 = 1.0 - fabs(pixels[i][j+1].R - pixels[i][j].R);
					wR3 = 1.0 - fabs(pixels[i][j-1].R - pixels[i][j].R);
					
					wR = wR0 + wR1 + wR2 + wR3;
					
					wR0 /= wR; wR1 /= wR; wR2 /= wR; wR3 /= wR;
					
					wG0 = 1.0 - fabs(pixels[i+1][j].G - pixels[i][j].G);
					wG1 = 1.0 - fabs(pixels[i-1][j].G - pixels[i][j].G);
					wG2 = 1.0 - fabs(pixels[i][j+1].G - pixels[i][j].G);
					wG3 = 1.0 - fabs(pixels[i][j-1].G - pixels[i][j].G);
					
					wG = wG0 + wG1 + wG2 + wG3;
					
					wG0 /= wG; wG1 /= wG; wG2 /= wG; wG3 /= wG;
					
					wB0 = 1.0 - fabs(pixels[i+1][j].B - pixels[i][j].B);
					wB1 = 1.0 - fabs(pixels[i-1][j].B - pixels[i][j].B);
					wB2 = 1.0 - fabs(pixels[i][j+1].B - pixels[i][j].B);
					wB3 = 1.0 - fabs(pixels[i][j-1].B - pixels[i][j].B);
					
					wB = wB0 + wB1 + wB2 + wB3;
					
					wB0 /= wB; wB1 /= wB; wB2 /= wB; wB3 /= wB;
					
					pixels[i][j].R = wR0*pixels[i+1][j].R + wR1*pixels[i-1][j].R +
										wR2*pixels[i][j+1].R + wR3*pixels[i][j-1].R;
										
					pixels[i][j].G = wG0*pixels[i+1][j].G + wG1*pixels[i-1][j].G +
										wG2*pixels[i][j+1].G + wG3*pixels[i][j-1].G;
										
					pixels[i][j].B = wB0*pixels[i+1][j].B + wB1*pixels[i-1][j].B +
										wB2*pixels[i][j+1].B + wB3*pixels[i][j-1].B;
			
			}
		}
		steps++;	
	}
}

// computes theta_i using the proper vectors
// mind that, by the dot product, A . B = |A| |B| cos (theta)
// theta_r = theta_i
double compute_theta_i(double *N_hit, ray* RAY){
	double theta_i, theta_r;
	double v0[3] = {0.0, 0.0, 0.0}; // auxiliar to compute magnitude of array using distance function
	double I[3] = {-RAY->dx, -RAY->dy, -RAY->dz}; // vector of incident ray
	return acos(dot(N_hit, I)/(distance(N_hit, v0)*distance(I, v0)));
}

// computes theta_t using Snell's Law, assuming TIR is not true in the main routine
double compute_theta_t(double n1, double n2, double theta_i){
	return asin(n1*sin(theta_i)/n2);
}

// checks for total reflection (not necessarily internal, can be used for external too)
bool TIR(double theta_i, double n1, double n2){
	if (theta_i == 0.0) return false;
	else if (sin(theta_i) > (n2/n1)) return true;
	else return false;
}

// Schlick's Approximation to compute Reflectance
double R_sch(double n1, double n2, double theta_i, double theta_t, bool TIR){
	double R0 = pow((n1 - n2)/(n1 + n2), 2);
	// if the pow is an integer, in C ,it is probably the same speed as doing 5 multiplications
	if (n1 > n2 && TIR == true){
		return 1.0;
	}
	else if (n1 > n2 && TIR == false){
		return R0 + (1.0 - R0)*pow(1.0 - cos(theta_t), 5);
	}
	else if(n1 <= n2){
		return R0 + (1.0 - R0)*pow(1.0 - cos(theta_i), 5);
	}
}

// computed transmitted vector based on vector form of Snell's Law
void compute_transmitted(double *t, double n1, double n2, double *inc, double *N, double theta_i, double theta_t, bool outside){
	double A, B;
	A = n1/n2;
	B = A*cos(theta_i) - cos(theta_t);

	//if (outside == false && cos(theta_i) < 0.0) printf("[!] WARNING\n");
	
	if(outside == false){
		N[0] = -N[0];
		N[1] = -N[1];
		N[2] = -N[2];
	}
	
	t[0] = A*inc[0] + B*N[0];
	t[1] = A*inc[1] + B*N[1];
	t[2] = A*inc[2] + B*N[2];
	
	norm_vector(t);
}

/*
void compute_transmitted(double *t, double n1, double n2, double *inc, double *N, double theta_i, double theta_t, bool outside){
    double ratio = n1 / n2;
    double cos_theta_i = cos(theta_i);
    double sin_theta_t = sin(theta_t);

    // Calculate the transmitted vector components
    t[0] = ratio * inc[0] + (ratio * cos_theta_i - sqrt(1.0 - pow(sin_theta_t, 2))) * N[0];
    t[1] = ratio * inc[1] + (ratio * cos_theta_i - sqrt(1.0 - pow(sin_theta_t, 2))) * N[1];
    t[2] = ratio * inc[2] + (ratio * cos_theta_i - sqrt(1.0 - pow(sin_theta_t, 2))) * N[2];

    // Normalize the transmitted vector
    norm_vector(t);
}
*/

// The Reflected ray needs a closest object intersection check, the Transmitted not, but could be used
// to compute internal intersection point inside sphere
// It works perfectly if objects change position instead of camera. If camera changes position, 
// everything else will be distorted to the previous camera position, would need to handle that
// properly.
color recursive_trace(ray primeray, double *light, int depth, int max_depth, int N_obj, sphere *s, double *inc_o, double *A, double *Im, bool outside){
	color C;
	C.R = A[0]; C.G = A[1]; C.B = A[2];
	// no contribition added at max depth
	if (depth >= max_depth) return C; // Terminate recursion if max depth is reached
	
	double min = INFINITY;
	double p_hit[3], N_hit[3];
	sphere *i_obj = NULL;
	for (int obj = 0; obj < N_obj; obj++){
		// check for intersection and compute N_hit and p_hit
		if (intersect(&s[obj], &primeray, p_hit, N_hit, outside)){
			double dist = distance(inc_o, p_hit);
			if (dist < min){
				min = dist;
				i_obj = &s[obj];
			}
		}
	}
	
    // Process intersection
    if (i_obj){
    
    	//
    	// Shading section
    	//
    	ray shadow;
		bool is_shadow;
		double abs_s;
		shadow.dx = light[0] - p_hit[0];
		shadow.dy = light[1] - p_hit[1];
		shadow.dz = light[2] - p_hit[2];
								
		// normalization
		abs_s = sqrt(shadow.dx*shadow.dx +
			 	 	shadow.dy*shadow.dy +
				 	shadow.dz*shadow.dz);
								
		shadow.dx /= abs_s;
		shadow.dy /= abs_s;
		shadow.dz /= abs_s;
								
		// slightly offset ray beggining along the hit normal in order to avoid precision problem
		double d_s[3] = {shadow.dx, shadow.dy, shadow.dz};
		//if (dot(d_s, N_hit) < 0) {
		if (outside == true){
			shadow.ox = p_hit[0] + N_hit[0]*OFFSET;
			shadow.oy = p_hit[1] + N_hit[1]*OFFSET;
			shadow.oz = p_hit[2] + N_hit[2]*OFFSET;
		}
		else {
			shadow.ox = p_hit[0] - N_hit[0]*OFFSET;
			shadow.oy = p_hit[1] - N_hit[1]*OFFSET;
			shadow.oz = p_hit[2] - N_hit[2]*OFFSET;
		}
				
		is_shadow = false;
				
		// iterates over higher order objects to see if it is in shadow area
		for (int hi_obj = 0; hi_obj < N_obj; hi_obj++){
			if (intersect(&s[hi_obj], &shadow, p_hit, N_hit, outside)){
				is_shadow = true;
				break;
			}
		}
				
		if (!(is_shadow)){
			color aux;
			aux = rho(i_obj, A, Im, &shadow, N_hit, &primeray);
			C.R += aux.R;
			C.G += aux.G;
			C.B += aux.B;
		}
			
    	//
    	// Reflection section
    	//
    	
    	ray ref; // ray tra is created if and only if TIR != 0 for deeper reflections
		double ref_d[3], inc_d[3], RR, RT, n1, n2;
							
		inc_d[0] = primeray.dx;
		inc_d[1] = primeray.dy;
		inc_d[2] = primeray.dz;
		
		norm_vector(inc_d);
							
		compute_reflected(ref_d, inc_d, N_hit); // need to know to compute transmitted too
		
		ref.dx = ref_d[0];
		ref.dy = ref_d[1];
		ref.dz = ref_d[2];
		
		// process origin of reflected ray
		//if (dot(ref_d, N_hit) < 0){
		if (outside == true){
			ref.ox = p_hit[0] + N_hit[0]*OFFSET;
			ref.oy = p_hit[1] + N_hit[1]*OFFSET;
			ref.oz = p_hit[2] + N_hit[2]*OFFSET;
		}
		else {
			ref.ox = p_hit[0] - N_hit[0]*OFFSET;
			ref.oy = p_hit[1] - N_hit[1]*OFFSET;
			ref.oz = p_hit[2] - N_hit[2]*OFFSET;
		}

		inc_o[0] = ref.ox;
		inc_o[1] = ref.oy;
		inc_o[2] = ref.oz;
    	
        // Adjust refractive indices based on whether the ray is entering or exiting the material
        //if (dot(inc_d, N_hit) > 0) {
        if (outside == true) {
           	n1 = n_AIR;
           	n2 = i_obj->M.n[0];
        } else {
           	n1 = i_obj->M.n[0];
           	n2 = n_AIR;
  		}
  		
  		double theta_i = compute_theta_i(N_hit, &primeray);
  		double theta_t = compute_theta_t(n1, n2, theta_i);
  		bool TST = TIR(theta_i, n1, n2);
  		
  		RR = R_sch(n1, n2, theta_i, theta_t, TST);
		RT = 1.0 - RR;
        // Calculate reflected color contribution
        color aux_R = recursive_trace(ref, light, depth + 1, max_depth, N_obj, s, inc_o, A, Im, outside);
		
        C.R += RR*aux_R.R;
        C.G += RR*aux_R.G;
        C.B += RR*aux_R.B;
        
        /*
        C.R += i_obj->c.R*RR*aux_R.R;
        C.G += i_obj->c.G*RR*aux_R.G;
        C.B += i_obj->c.B*RR*aux_R.B;
  		*/
  		//
  		// Refraction section
  		//
  		
  		//if (TST == false) printf("FALSE\n");
  		
  		if (i_obj->M.n[1] == 0.0 && TST == false){
            ray tra;
            double tra_d[3];
            color aux_T;
           
            compute_transmitted(tra_d, n1, n2, inc_d, N_hit, theta_i, theta_t, outside);
			
            tra.dx = tra_d[0];
            tra.dy = tra_d[1];
            tra.dz = tra_d[2];
           	
           	// process origin of transmitted ray
           	//	if (dot(tra_d, N_hit) < 0){
           	if (outside == true){
            	tra.ox = p_hit[0] - N_hit[0]*OFFSET;
            	tra.oy = p_hit[1] - N_hit[1]*OFFSET;
            	tra.oz = p_hit[2] - N_hit[2]*OFFSET;
            }
            else{
            	tra.ox = p_hit[0] + N_hit[0]*OFFSET;
           		tra.oy = p_hit[1] + N_hit[1]*OFFSET;
            	tra.oz = p_hit[2] + N_hit[2]*OFFSET;
            }

            inc_o[0] = tra.ox;
			inc_o[1] = tra.oy;
			inc_o[2] = tra.oz;
               	
			aux_T = recursive_trace(tra, light, depth + 1, max_depth, N_obj, s, inc_o, A, Im, !outside);
					
            // Calculate refracted color contribution
            /*
            C.R += RT*aux_T.R;
            C.G += RT*aux_T.G;
            C.B += RT*aux_T.B;
            */
            
            C.R += i_obj->c.R*RT*aux_T.R;
            C.G += i_obj->c.G*RT*aux_T.G;
            C.B += i_obj->c.B*RT*aux_T.B;
	
        }
    }
	return C;
}

// -------- main --------- //

int main(int argc, char *argv[]){
	if (argc < 2){
		printf("\n[!] resolution not provided. Will not run.\n\n");
		return -1;
	}
	
	double Lx = 5.0; // screen size in x
	double Ly = 5.0; // screen size in y
	int PIX_x = atoi(argv[1]); // number of pixels in x
	int PIX_y = atoi(argv[1]); // number of pixels in y
	double n_air = 1.0;
	
	// Dynamically allocate pixels array
	
    color **pixels = (color **)malloc(PIX_x*sizeof(color*));
    for (int i = 0; i < PIX_x; i++) {
        pixels[i] = (color *)malloc(PIX_y*sizeof(color));
    }
    
    // screen
    
	double dx, dy; // for positions in the screen
	double eye[3] = {0.0, 0.0, -2.1}; // eye (camera) position
	double screen[3] = {0.0, 0.0, 0.0}; // center of screen position
	
	// light: will work with 1 light (for now)
	
	double light[3] = {5.0, 0.0, 3.5}; // light source position
	double Im[3] = {1.0, 1.0, 1.0}; // RGB for infalling light
	double A[3] = {0.005, 0.005, 0.005}; // RGB for ambient light
	
	// create array of spheres
	
	int N_obj = 5;
	sphere *s = (sphere *)malloc(N_obj*sizeof(sphere)); // array with 2 spheres
	
	// create properties
	
	color white, indigo, shock, aqua, a_metal, a_wood, d_metal, d_wood, s_metal, s_wood, a_plastic, d_plastic, s_plastic, a_glass, d_glass, s_glass, a_diamond, d_diamond, s_diamond;
	init_color(&white, 1.0, 1.0, 1.0);
	init_color(&indigo, 0.5, 0.0, 1.0);
	init_color(&shock, 1.0, 0.0, 0.8);
	init_color(&aqua, 0.0, 1.0, 0.6);
	
	init_color(&a_metal, 0.2, 0.2, 0.2);
	init_color(&d_metal, 0.5, 0.5, 0.5);
	init_color(&s_metal, 1.0, 1.0, 1.0);
	
	init_color(&a_wood, 0.05, 0.02, 0.01);
	init_color(&d_wood, 0.6, 0.3, 0.1);
	init_color(&s_wood, 0.1, 0.1, 0.1);
	
	init_color(&a_plastic, 0.1, 0.1, 0.1);
	init_color(&d_plastic, 0.7, 0.7, 0.7);
	init_color(&s_plastic, 0.9, 0.9, 0.9);
	
	init_color(&a_glass, 0.1, 0.1, 0.1);
	init_color(&d_glass, 0.0, 0.0, 0.0);
	init_color(&s_glass, 1.0, 1.0, 1.0);
	
	init_color(&a_diamond, 0.1, 0.1, 0.1);
	init_color(&d_diamond, 0.0, 0.0, 0.0);
	init_color(&s_diamond, 1.0, 1.0, 1.0);
	
	material metal, wood, plastic, glass, diamond;
	init_material(&metal, a_metal, d_metal, s_metal, 1.0, 128.0, 0.05, 3.5);
	init_material(&wood, a_wood, d_wood, s_wood, 0.0, 20.0, 1.3, 1.0);
	init_material(&plastic, a_plastic, d_plastic, s_plastic, 0.3, 32.0, 1.49, 1.0);
	init_material(&glass, a_glass, d_glass, s_glass, 0.0, 128.0, 1.4, 0.0);
	init_material(&diamond, a_diamond, d_diamond, s_diamond, 0.0, 380.0, 2.5, 0.0);
	
	// initialize N_obj spheres
	
	init_sphere(&s[0], 1.0, 0.0, 9.0, 3.0, 1, indigo, metal);
	init_sphere(&s[1], -1.0, -3.0, 3.5, 0.5, 0, aqua, plastic);
	init_sphere(&s[2], 0.0, 3.0, 3.0, 0.5, 2, shock, wood);
	init_sphere(&s[3], 0.5, 0.59, 0.3, 0.3, 3, white, glass);
	init_sphere(&s[4], -0.3, -0.15, 0.3, 0.3, 4, aqua, diamond);
	
	dx = Lx/PIX_x; // for pixels in x
	dy = Ly/PIX_y; // for pixels in y
	
	// initialize pixels as ambient light color
	for (int i = 0; i < PIX_x; i++){
		for (int j = 0; j < PIX_y; j++){
			pixels[i][j].R = A[0];
			pixels[i][j].G = A[1];
			pixels[i][j].B = A[2];
		}
	}
	
	double inc_o[3] = {eye[0], eye[1], eye[2]};
	#pragma omp parallel for collapse(2) private(inc_o)
	for (int i = 0; i < PIX_x; i++){
		for (int j = 0; j < PIX_y; j++){
			ray primeray;
			compute_prime(&primeray, eye, (i + 0.5)*dx - 0.5*Lx + screen[0], (j + 0.5)*dy - 0.5*Ly + screen[1], screen[2]);
			pixels[i][j] = recursive_trace(primeray, light, 0, 20, N_obj, s, inc_o, A, Im, true);
		}
	}
	
	// apply aaf filter
	aaf(pixels, PIX_x, PIX_y, 1);
	
	// print values to file
	FILE *pix_file = fopen("shaded.txt","w");
	for (int i = 0; i < PIX_x; i++){
		for (int j = 0; j < PIX_y; j++){
			fprintf(pix_file, "%d %d %lf %lf %lf\n", i, j, pixels[i][j].R, pixels[i][j].G, pixels[i][j].B);
		}
	}
	fclose(pix_file);
	
	free(s);
	free(pixels);
	
	system("python3 plot.py");
	
	return 0;
}
