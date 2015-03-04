#include <helper_image.h>     // helper for image and data compariosn
#include <helper_cuda.h>      // helper for cuda error checking functions


#define IX(i,j) ((i)+(N+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define FOR_EACH_CELL for ( i=1 ; i<=N ; i++ ) { for ( j=1 ; j<=N ; j++ ) {
#define END_FOR }}

#define Nsize 32
extern float * dev_a;
extern float * dev_b;
extern int sz;
#define ITERATIONS 100

__global__ void cuda_add_source ( int N, float * x, float * s, float dt )
{
	int size=(N+2)*(N+2);
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x_cpy = x[i];
	if (i < size)    // gridDim should be at least <<<N+2, N+2>>>
		x_cpy += dt*s[i];     // reuse of x in one thread; s stay in global memory could be coalesced.
	x[i] = x_cpy;
}

void add_source ( int N, float * x, float * s, float dt )
{
	int i, size=(N+2)*(N+2);
	for ( i=0 ; i<size ; i++ ) x[i] += dt*s[i];
}

__global__ void cuda_add_buoyancy( int N, float * dens, float * v, float dt, float by)
{
	int size=(N+2)*(N+2);
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float v_cpy = v[i];
	if (i < size)    // gridDim should be at least <<<N+2, N+2>>>
                v_cpy += -dens[i]*by*dt;     // reuse of v in one thread.
	v[i] = v_cpy;
}

void add_buoyancy( int N, float * dens, float * v, float dt, float by) 
{
        int i, size=(N+2)*(N+2);
        for ( i=0; i<size; i++ ) v[i] += -dens[i]*by*dt;
}

void set_bnd ( int N, int b, float * x )
{
	int i;

	for ( i=1 ; i<=N ; i++ ) {
		x[IX(0  ,i)] = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
		x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
		x[IX(i,0  )] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
		x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
	}
	x[IX(0  ,0  )] = 0.5f*(x[IX(1,0  )]+x[IX(0  ,1)]);
	x[IX(0  ,N+1)] = 0.5f*(x[IX(1,N+1)]+x[IX(0  ,N)]);
	x[IX(N+1,0  )] = 0.5f*(x[IX(N,0  )]+x[IX(N+1,1)]);
	x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)]+x[IX(N+1,N)]);
}

__device__  void device_set_bnd(int N, int b, float* x, int i, int j)
{
        if(i == 0)
                x[IX(i, j)] = b==1 ? -x[IX(i+1, j)]:x[IX(i+1, j)];
        if(i == N+1)
                x[IX(i, j)] = b==1 ? -x[IX(i-1, j)]:x[IX(i-1, j)];
        if(j == 0)
                x[IX(i, j)] = b==2 ? -x[IX(i, j+1)]:x[IX(i, j+1)];
        if(j == N+1)
                x[IX(i, j)] = b==2 ? -x[IX(i, j-1)]:x[IX(i, j-1)];

        if(i == 0 && j == 0)
                x[IX(i, j)] = 0.5*(x[IX(i+1, j)] + x[IX(i, j+1)]);
        if(i == 0 && j == N+1)
                x[IX(i, j)] = 0.5*(x[IX(i+1, j)] + x[IX(i, j-1)]);
        if(i == N+1 && j == 0)
                x[IX(i, j)] = 0.5*(x[IX(i-1, j)] + x[IX(i, j+1)]);
        if(i == N+1 && j == N+1)
                x[IX(i, j)] = 0.5*(x[IX(i-1, j)] + x[IX(i, j-1)]);
}

void vorticity(int N, int dt, float * u, float * v, float * u0, float * v0, float *vort, float *vl, float *vfx, float *vfy )
{
        int i, j;
        float halfrdx = 0.5 * N;
        float len;

        for (i = 1; i <= N; i++) {
                for (j = 1; j <= N; j++) {
                        vort[IX(i, j)] = halfrdx * ((v[IX(i + 1, j)] - v[IX(i - 1, j)]) - (u[IX(i, j + 1)] - u[IX(i, j - 1)]));
                        vl[IX(i, j)] = fabs(vort[IX(i, j)]);
                }
        }
        set_bnd(N, 0, vort);
        set_bnd(N, 0, vl);

        for (i = 1; i <= N; i++) {
                for (j = 1; j <= N; j++) {
                        vfx[IX(i, j)] = halfrdx * (vl[IX(i + 1, j)] - vl[IX(i - 1, j)]);
                        vfy[IX(i, j)] = halfrdx * (vl[IX(i, j + 1)] - vl[IX(i, j - 1)]);
                        len = sqrt(vfx[IX(i, j)] * vfx[IX(i, j)] + vfy[IX(i, j)] * vfy[IX(i, j)]);
                        if (len < 1e-10) {
                                vfx[IX(i, j)] = 0.0f;
                                vfy[IX(i, j)] = 0.0f;
                        }
                        else {
                                vfx[IX(i, j)] /= len;
                                vfy[IX(i, j)] /= len;
                        }
                }
        }
        set_bnd(N, 0, vfx);
        set_bnd(N, 0, vfy);

        for (i = 1; i <= N; i++) {
                for (j = 1; j <= N; j++) {
                        u0[IX(i, j)] += dt * 0.01f * (vfy[IX(i, j)] * vort[IX(i, j)]);
                        v0[IX(i, j)] += dt * 0.01f * (-vfx[IX(i, j)] * vort[IX(i, j)]);
                }
        }
}

__global__ void cuda_vorticity(int N, int dt, float * u, float * v, float * u0, float * v0, float *vl)
{
	int i, j;
        float halfrdx = 0.5 * N;
        float len;
	float vort_reg, vfx_reg, vfy_reg;
	i = threadIdx.x + blockIdx.x*blockDim.x+1;
        j = threadIdx.y + blockIdx.y*blockDim.y+1;

	vort_reg = halfrdx * ((v[IX(i + 1, j)] - v[IX(i - 1, j)]) - (u[IX(i, j + 1)] - u[IX(i, j - 1)]));
        vl[IX(i, j)] = fabs(vort_reg);

        vfx_reg = halfrdx * (vl[IX(i + 1, j)] - vl[IX(i - 1, j)]);
        vfy_reg = halfrdx * (vl[IX(i, j + 1)] - vl[IX(i, j - 1)]);
        len = sqrt(vfx_reg * vfx_reg + vfy_reg * vfy_reg);
        if (len < 1e-10) {
                vfx_reg = 0.0f;
                vfy_reg = 0.0f;
        }
        else {
                vfx_reg /= len;
                vfy_reg /= len;
        }

        u[IX(i, j)] += dt * 0.01f * (vfy_reg * vort_reg);
        v[IX(i, j)] += dt * 0.01f * (-vfx_reg * vort_reg);
}

void lin_solve ( int N, int b, float * x, float * x0, float a, float c )
{
	int i, j, k;

	for ( k=0 ; k<20 ; k++ ) {
		FOR_EACH_CELL
			x[IX(i,j)] = (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/c;
		END_FOR
		set_bnd ( N, b, x );
	}
}

__global__ void cuda_lin_solve ( int N, int b, float * x, float * x0, float a, float c )
{
	//global mem vesion
	//int N=Nsize;
	int i = blockIdx.x * Nsize + threadIdx.x+1;//i j  vs   j i
	int j = blockIdx.y * Nsize + threadIdx.y+1;

	for (int  k=0 ; k<ITERATIONS ; k++ ) {
		//float temp=0;
		if((i+j)%2==0)
		{
			x[IX(i,j)]= (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/c;
		}
		__syncthreads();
		if((i+j)%2==1)
		{
			x[IX(i,j)]= (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/c;
					
		}
		
		__syncthreads();
		if(i+j==2)
		{
		
			int ii;
			

			for ( ii=1 ; ii<=N ; ii++ ) 
			{
				x[IX(0  ,ii)] = b==1 ? -x[IX(1,ii)] : x[IX(1,ii)];
				x[IX(Nsize+1,ii)] = b==1 ? -x[IX(Nsize,ii)] : x[IX(Nsize,ii)];
				x[IX(ii,0  )] = b==2 ? -x[IX(ii,1)] : x[IX(ii,1)];
				x[IX(ii,Nsize+1)] = b==2 ? -x[IX(ii,Nsize)] : x[IX(ii,Nsize)];
			}
			x[IX(0  ,0  )] = 0.5f*(x[IX(1,0  )]+x[IX(0  ,1)]);
			x[IX(0  ,Nsize+1)] = 0.5f*(x[IX(1,Nsize+1)]+x[IX(0  ,Nsize)]);
			x[IX(Nsize+1,0  )] = 0.5f*(x[IX(Nsize,0  )]+x[IX(Nsize+1,1)]);
			x[IX(Nsize+1,Nsize+1)] = 0.5f*(x[IX(Nsize,Nsize+1)]+x[IX(Nsize+1,Nsize)]);
		}
		__syncthreads();
		// set_bnd ( N, b, x );
		//set_bndcuda ( N, b, x );
	};	
}


void diffuse ( int N, int b, float * x, float * x0, float diff, float dt )
{
	float a=dt*diff*N*N;
    //printf("a: %f \n", a);
    dim3 Grid3	((double)(N)/32.0, (double)(N)/32.0); 
    dim3 Block3	(32,32);
	

    cudaMemcpy (dev_a , x ,      sz*sizeof(float) , cudaMemcpyHostToDevice);
	cudaMemcpy (dev_b , x0 , sz*sizeof(float) , cudaMemcpyHostToDevice);

	cuda_lin_solve<<<Grid3, Block3>>>( N, b, dev_a, dev_b, a, 1+4*a );
	
	cudaMemcpy (x ,dev_a ,      sz*sizeof(float) , cudaMemcpyDeviceToHost);
	cudaMemcpy (x0 ,dev_b,  sz*sizeof(float) , cudaMemcpyDeviceToHost);	

}

void cuda_diffuse ( int N, int b, float * x, float * x0, float diff, float dt )
{

}


void advect ( int N, int b, float * d, float * d0, float * u, float * v, float dt )
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	dt0 = dt*N;
	FOR_EACH_CELL
		x = i-dt0*u[IX(i,j)]; y = j-dt0*v[IX(i,j)];
		if (x<0.5f) x=0.5f; if (x>N+0.5f) x=N+0.5f; i0=(int)x; i1=i0+1;
		if (y<0.5f) y=0.5f; if (y>N+0.5f) y=N+0.5f; j0=(int)y; j1=j0+1;
		s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
		d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j1)])+
					 s1*(t0*d0[IX(i1,j0)]+t1*d0[IX(i1,j1)]);
	END_FOR
	set_bnd ( N, b, d );
}

void project_pre( int N, float * u, float * v, float * p, float * div )
{

}

void project ( int N, float * u, float * v, float * p, float * div )
{
	int i, j;

	FOR_EACH_CELL
		div[IX(i,j)] = -0.5f*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)])/N;
		p[IX(i,j)] = 0;
	END_FOR	
	set_bnd ( N, 0, div ); set_bnd ( N, 0, p );

//	if(1==0)
//		lin_solve ( N, 0, p, div, 1, 4 );
//	else{


		dim3 Grid3  ((double)(N)/32.0, (double)(N)/32.0);
		dim3 Block3 (32,32,1);
		
//		printf("project CU bef %f %f %f \n", u[IX(3,3)],u[IX(3,4)],u[IX(3,5)]);
		
		cudaMemcpy(dev_a, p, sz*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, div, sz*sizeof(float), cudaMemcpyHostToDevice);
		
		cuda_lin_solve<<<Grid3, Block3>>>( N, 0, dev_a, dev_b, 1, 4 );
		
		cudaMemcpy(p, dev_a, sz*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(div, dev_b, sz*sizeof(float), cudaMemcpyDeviceToHost); 
		
//		printf("project CU aft %f %f %f \n", u[IX(3,3)],u[IX(3,4)],u[IX(3,5)]);



//		};
	
	FOR_EACH_CELL
		u[IX(i,j)] -= 0.5f*N*(p[IX(i+1,j)]-p[IX(i-1,j)]);
		v[IX(i,j)] -= 0.5f*N*(p[IX(i,j+1)]-p[IX(i,j-1)]);
	END_FOR
	set_bnd ( N, 1, u ); set_bnd ( N, 2, v );
}

void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt, float by )
{
	add_source ( N, x, x0, dt );
	SWAP ( x0, x ); diffuse ( N, 0, x, x0, diff, dt );
	SWAP ( x0, x ); advect ( N, 0, x, x0, u, v, dt );
}
__global__ void swap(int N, float* a, float* b)
{
	int i = blockIdx.x * Nsize + threadIdx.x+1;//i j  vs   j i
	int j = blockIdx.y * Nsize + threadIdx.y+1;

	float temp = a[IX(i, j)];
	a[IX(i, j)] = b[IX(i, j)];
	b[IX(i, j)] = temp;


}
//void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt, float by, float * dens )
__host__ void test ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt, float by, float * dens, float *vort, float *vl, float *vfx, float *vfy )
{
//ini
static int inimem=0;
if(inimem==0)
{
        inimem++;
        cudaMalloc((void **)&dev_a , sz*sizeof(float)  );  
        cudaMalloc((void **)&dev_b , sz*sizeof(float)  );  
        //cudaMalloc((void **)&dev_c , size*sizeof(float)  );
}
	float a=dt*visc*N*N;
	int size = (N+2)*(N+2);  int numbytes = size*sizeof(float);
	float * cuda_u; cudaMalloc((void **)&cuda_u, numbytes); cudaMemcpy(cuda_u, u, numbytes, cudaMemcpyHostToDevice);
	float * cuda_v; cudaMalloc((void **)&cuda_v, numbytes); cudaMemcpy(cuda_v, v, numbytes, cudaMemcpyHostToDevice);
	float * cuda_u0; cudaMalloc((void **)&cuda_u0, numbytes); cudaMemcpy(cuda_u0, u0, numbytes, cudaMemcpyHostToDevice);
	float * cuda_v0; cudaMalloc((void **)&cuda_v0, numbytes); cudaMemcpy(cuda_v0, v0, numbytes, cudaMemcpyHostToDevice);
	float * temp; cudaMalloc((void **)&temp, numbytes);
	float * cuda_dens; cudaMalloc((void **)&cuda_dens, numbytes); cudaMemcpy(cuda_dens, dens, numbytes, cudaMemcpyHostToDevice);

	float * cuda_vl; cudaMalloc((void **)&cuda_vl, numbytes); cudaMemcpy(cuda_vl, vl, numbytes, cudaMemcpyHostToDevice);

dim3 dimGrid(N+2, 1);
dim3 dimBlock(N+2, 1);
	cuda_add_source<<<dimGrid, dimBlock>>>(N, cuda_u, cuda_u0, dt); 
	cuda_add_source<<<dimGrid, dimBlock>>>(N, cuda_v, cuda_v0, dt); 
	cuda_add_buoyancy<<<dimGrid, dimBlock>>>(N, cuda_dens, cuda_v, dt, by);

dim3 threadPerBlock(32, 32);
dim3 blockPerGrid((double)(N)/32.0, (double)(N)/32.0);
	//cuda_vorticity<<<blockPerGrid, threadPerBlock>>>(N, dt, cuda_u, cuda_v, cuda_u0, cuda_v0, cuda_vl);

/********************************/
	//cudaDeviceSynchronize();

	swap<<<blockPerGrid, threadPerBlock>>>(N, cuda_u0, cuda_u);

	//SWAP ( cuda_u0, cuda_u );
    //cudaDeviceSynchronize();
//	cudaMemcpy(temp, cuda_u, numbytes, cudaMemcpyDeviceToDevice);
//	cudaMemcpy(cuda_u, cuda_u0, numbytes, cudaMemcpyDeviceToDevice);
//	cudaMemcpy(cuda_u0, temp, numbytes, cudaMemcpyDeviceToDevice);
	cuda_lin_solve<<<blockPerGrid, threadPerBlock>>>( N, 1, cuda_u, cuda_u0, a, 1+4*a );
	///SWAP( cuda_u0, cuda_u);
	//cudaDeviceSynchronize();
     //printf("before: %p, %p", cuda_v0, cuda_v);
	//SWAP ( cuda_v0, cuda_v );
     //printf("after: %p, %p", cuda_v0, cuda_v);
     //getchar();
     //cudaDeviceSynchronize();
	swap<<<blockPerGrid, threadPerBlock>>>(N, cuda_v0, cuda_v);
	//	cudaMemcpy(temp, cuda_v, numbytes, cudaMemcpyDeviceToDevice);
//	cudaMemcpy(cuda_v, cuda_v0, numbytes, cudaMemcpyDeviceToDevice);
//	cudaMemcpy(cuda_v0, temp, numbytes, cudaMemcpyDeviceToDevice);

	cuda_lin_solve<<<blockPerGrid, threadPerBlock>>>( N, 2, cuda_v, cuda_v0, a, 1+4*a );
	//SWAP(cuda_v0, cuda_v);
	// cudaDeviceSynchronize();

/********************************/
//dim3 Grid3	(1,1,1); 
//dim3 Block3	(N,N,1);

	cudaMemcpy(u, cuda_u, numbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, cuda_v, numbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(u0, cuda_u0, numbytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(v0, cuda_v0, numbytes, cudaMemcpyDeviceToHost);

	cudaFree(cuda_u); cudaFree(cuda_v); cudaFree(cuda_u0); cudaFree(cuda_v0); cudaFree(cuda_dens);
	cudaFree(cuda_vl);

}
void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt, float by, float * dens, float *vort, float *vl, float *vfx, float *vfy )
{
	test(N, u, v, u0, v0, visc, dt, by, dens, vort, vl, vfx, vfy );
	//SWAP ( u0, u );
	//diffuse ( N, 1, u, u0, visc, dt );
	//SWAP(u0, u);
	//SWAP ( v0, v );
	//diffuse ( N, 2, v, v0, visc, dt );
	//SWAP(v0, v);
	project ( N, u, v, u0, v0 );
	SWAP ( u0, u ); SWAP ( v0, v );
	advect ( N, 1, u, u0, u0, v0, dt ); advect ( N, 2, v, v0, u0, v0, dt );
	project ( N, u, v, u0, v0 );
}
