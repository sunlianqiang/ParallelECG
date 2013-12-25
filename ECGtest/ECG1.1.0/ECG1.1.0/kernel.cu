/********************************************************************
*  sample.cu
*  This is a example of the CUDA program.

cuda version < 5.0 error

*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


//5.0版本的Cuda不支持cutil.h 文件
//最保险的是把原程序里用到的 CUDA_SAFE_CALL之类的宏都自己实现一遍，简单的话可以从原本的cutil.h里复制出来，以后都可以不用依赖这个文件了
//你看看CUDA SDK中的那些示例程序，现在纠错都是用checkCudaErrors函数,使用的头文件是help_cuda.h
//CUDA_SAFE_CALL(..) is replaced by checkCudaErrors(..) which is available in helper_cuda.h file. 
// #include <cutil.h>
// #include <cutil_inline.h>
//#include <helper_cuda.h>

const short int ND = 1;
const short int ND3 = 1;
const short int NI = 56;
const short int NJ = 56;
const short int NK = 90;
const short int NL = 344;
const short int NPARM = 35;
const short int NCELL = 14;
const short int INFTIME = 9999;
const short int ANISO = 1; /*<Comment by ALF> aniso switch*/
const short int NCYCL = 20; /*<Comment by ALF> max cycle num*/
const short int TSTEP = 2000;
const short int NENDO = 4000;
const short int Nepic=NI*NJ*2;

float *d_r,*d_rn,*d_tm;
short int *d_tnd;
float *d_POTi=0, *d_der=0,*d_endoHnnA=0,*d_surfPOTi=0;
short int *d_endoBx=0;
short int *d_endoBy=0;
short int *d_endoBz=0;
short int *d_endoCx=0;
short int *d_endoCy=0;
short int *d_endoCz=0;

short int *d_epicX=0;
short int *d_epicY=0;
short int *d_epicZ=0;
float *d_epicPOTold=0;

extern "C" void hpc();
extern "C" void gpu_freetransdata();
extern "C" void gpu_transdata(short int epicX[Nepic],short int epicY[Nepic],short int epicZ[Nepic],short int *g_tnd[3],float *g_r[3],float *g_rn[3],short int g_endoBx[NENDO*ND3],short int g_endoBy[NENDO*ND3],short int g_endoBz[NENDO*ND3],short int g_endoCx[NENDO*ND3],short int g_endoCy[NENDO*ND3],short int g_endoCz[NENDO*ND3],float g_tm[3][6]);
extern "C" void gpu_BSPitmm_Malloc(float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi);
extern "C" void gpu_BSPitmm_HostToDevice(float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi);
extern "C" void gpu_BSPitmm_DeviceToHost(float *g_epicPOTold,float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi);

extern "C" void gpu_dpl_all(short int do_epicPOT,float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL],
							float g_HRTx0,float g_HRTy0,float g_HRTz0,int g_NendoB,int g_NendoC,
						float *g_endoHnnA,short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6],float *g_epicPOTold);

extern "C" void gpu_dpl_nPos(float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL]);
extern "C" void gpu_dpl_nPos_2(float g_posi,float g_posj,float g_posk,float g_dpl[3]);
extern "C" void gpu_dpl_Nendo(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
							  int g_NendoBC,int g_offset,float g_dpl[3],float *g_endoHnnA,
							  short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6]);
extern "C" void gpu_dpl_Nepic(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
							  float g_dpl[3],float g_tm[3][6],float *g_epicPOTold);


//extern "C" void dplpro(float *POTi,const short int NL, const float **r);


/************************************************************************/
/* Init CUDA                                                            */
/************************************************************************/
#if __DEVICE_EMULATION__

bool InitCUDA(void){return true;}

#else
bool InitCUDA(void)
{
	int count = 0;
	int i = 0;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}
	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA.\n");
		return false;
	}
	cudaSetDevice(i);

	printf("CUDA initialized.\n");
	return true;
}

#endif

/************************************************************************/
/* Example                                                              */
/************************************************************************/
__global__ static void k_dpl_Nepic(short int *k_epicX,short int *k_epicY,short int *k_epicZ,float k_posi,float k_posj,float k_posk,
								   float k_HRTx0,float k_HRTy0,float k_HRTz0,float *k_dpl,float *k_epicPOTold,
								   float *k_tm,short int k_Nepic)
{
float ax,ay,az,r1,r2,r3,dr,rv3,tmp1,tmp2,tmp3;
int n=blockDim.x * blockIdx.x + threadIdx.x;
if (n< k_Nepic)			
	{					//for (n=0; n<Nepic; ++n) {
						//ax=HRTx0+epicX[n]*tmswf[0][0]+epicY[n]*tmswf[0][1]+epicZ[n]*tmswf[0][2]-posi;
						//ay=HRTy0+epicX[n]*tmswf[1][0]+epicY[n]*tmswf[1][1]+epicZ[n]*tmswf[1][2]-posj;
						//az=HRTz0+epicX[n]*tmswf[2][0]+epicY[n]*tmswf[2][1]+epicZ[n]*tmswf[2][2]-posk;
		ax=k_HRTx0;
		tmp1=*(k_epicX+n) * *(k_tm);
		ax=ax+tmp1;
		tmp2=*(k_epicY+n) * *(k_tm+1);
		ax=ax+tmp2;
		tmp3=*(k_epicZ+n) * *(k_tm+2);
		ax=ax+tmp3;
		ax=ax-k_posi;
		ay=k_HRTy0;
		tmp1=*(k_epicX+n) * *(k_tm+1*6);
		ay=ay+tmp1;
		tmp2=*(k_epicY+n) * *(k_tm+1*6+1);
		ay=ay+tmp2;
		tmp3=*(k_epicZ+n) * *(k_tm+1*6+2);
		ay=ay+tmp3;
		ay=ay-k_posj;
		az=k_HRTz0;
		tmp1=*(k_epicX+n) * *(k_tm+2*6);
		az=az+tmp1;
		tmp2=*(k_epicY+n) * *(k_tm+2*6+1);
		az=az+tmp2;
		tmp3=*(k_epicZ+n) * *(k_tm+2*6+2);
		az=az+tmp3;
		az=az-k_posk;

		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		//dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
		tmp1=k_dpl[0]*ax;
		dr=tmp1;
		tmp2=k_dpl[1]*ay;
		dr+=tmp2;
		tmp3=k_dpl[2]*az;
		dr+=tmp3;

		rv3=1/r3;
		*(k_epicPOTold+n)+=dr*rv3;
	}
}
__global__ static void k_dpl_Nendo(float k_posi,float k_posj,float k_posk,
								   float k_HRTx0,float k_HRTy0,float k_HRTz0,int k_NendoB,int k_offset,float *k_dpl,
								   float *k_endoHnnA,short int *k_endoBx,short int *k_endoBy,short int *k_endoBz,
								   float *k_tm)
{
	float ax,ay,az,r1,r2,r3,dr,rv3,tmp1,tmp2,tmp3;
	int n=blockDim.x * blockIdx.x + threadIdx.x;
	if (n< k_NendoB)			
	{
		//ax=k_HRTx0+*(k_endoBx+n) * *(k_tm)+*(k_endoBy+n) * *(k_tm+1)+*(k_endoBz+n) * *(k_tm+2)-k_posi;
		//ay=k_HRTy0+*(k_endoBx+n) * *(k_tm+1*6)+*(k_endoBy+n) * *(k_tm+1*6+1)+*(k_endoBz+n) * *(k_tm+1*6+2)-k_posj;
		//az=k_HRTz0+*(k_endoBx+n) * *(k_tm+2*6)+*(k_endoBy+n) * *(k_tm+2*6+1)+*(k_endoBz+n) * *(k_tm+2*6+2)-k_posk;
		ax=k_HRTx0;
		tmp1=*(k_endoBx+n) * *(k_tm);
		ax=ax+tmp1;
		tmp2=*(k_endoBy+n) * *(k_tm+1);
		ax=ax+tmp2;
		tmp3=*(k_endoBz+n) * *(k_tm+2);
		ax=ax+tmp3;
		ax=ax-k_posi;
		ay=k_HRTy0;
		tmp1=*(k_endoBx+n) * *(k_tm+1*6);
		ay=ay+tmp1;
		tmp2=*(k_endoBy+n) * *(k_tm+1*6+1);
		ay=ay+tmp2;
		tmp3=*(k_endoBz+n) * *(k_tm+1*6+2);
		ay=ay+tmp3;
		ay=ay-k_posj;
		az=k_HRTz0;
		tmp1=*(k_endoBx+n) * *(k_tm+2*6);
		az=az+tmp1;
		tmp2=*(k_endoBy+n) * *(k_tm+2*6+1);
		az=az+tmp2;
		tmp3=*(k_endoBz+n) * *(k_tm+2*6+2);
		az=az+tmp3;
		az=az-k_posk;


		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		//dr=k_dpl[0]*ax+k_dpl[1]*ay+k_dpl[2]*az;
		tmp1=k_dpl[0]*ax;
		dr=tmp1;
		tmp2=k_dpl[1]*ay;
		dr+=tmp2;
		tmp3=k_dpl[2]*az;
		dr+=tmp3;

		rv3=1/r3;
		*(k_endoHnnA+k_offset+n)+=dr*rv3;
	};
}


__global__ static void k_dpl_nPos_2(float k_posi,float k_posj,float k_posk,float *k_dpl,float *k_r,float *d_surfPOTi,
									short int *d_tnd)
{
	float ax,ay,az,r1,r2,r3,dr,rv3;
	int n0,n1,n2;
	int n=blockDim.x * blockIdx.x + threadIdx.x;
	//if (n< ((NL-2)*2))			
	//{
		n0=d_tnd[n]-1;
		n1=d_tnd[(NL-2)*2+n]-1;
		n2=d_tnd[(NL-2)*2*2+n]-1;
		ax=(k_r[n0]+k_r[n1]+k_r[n2])/3-k_posi;
		ay=(k_r[NL+n0]+k_r[NL+n1]+k_r[NL+n2])/3-k_posj;
		az=(k_r[2*NL+n0]+k_r[2*NL+n1]+k_r[2*NL+n2])/3-k_posk;
		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		dr=ax;
		dr=dr*k_dpl[0];
		dr+=k_dpl[1]*ay;
		dr+=k_dpl[2]*az;
		rv3=1/r3;
		*(d_surfPOTi+n)+=dr*rv3;

	//};
}
__global__ void k_dpl_nPos(float k_posi,float k_posj,float k_posk,int k_nPos,float *k_dpl,
								  float *k_POTi,float *k_der,float *k_r ,float *k_rn )
{
	float ax,ay,az,r1,r2,r3,r5,dr,ds,rv3,bx,by,bz,ret_der,ret_POTi;
	int n=threadIdx.x;
	ax=k_r[n];
	ay=k_r[NL+n];
	az=k_r[2*NL+n];
	ax = ax - k_posi;
	ay = ay - k_posj;
	az = az - k_posk;

	r2=ax*ax+ay*ay+az*az;
	r1=(float)sqrt(r2);
	r3=(float)(r1*r2);
	r5=(float)(r2*r3);
	dr=k_dpl[0]*ax+k_dpl[1]*ay+k_dpl[2]*az;
	ds=3*dr/r5;
	rv3=1/r3;
	bx=k_dpl[0]*rv3-ax*ds;
	by=k_dpl[1]*rv3-ay*ds;
	bz=k_dpl[2]*rv3-az*ds;
		//*(k_der+n)+=*(d_rn[0]+n)*bx+*(d_rn[1]+n)*by+*(d_rn[2]+n)*bz;
	ret_der  = k_der[n];
	ret_der += k_rn[n]*bx;
	ret_der += k_rn[NL+n]*by;
	ret_der += k_rn[2*NL+n]*bz;
	k_der[n] = ret_der;

	//*(k_POTi+n)+=dr*rv3;
	ret_POTi = k_POTi[n];
	ret_POTi += dr*rv3;
	k_POTi[n] = ret_POTi;
	 //__syncthreads();
}
extern "C" void gpu_freetransdata()
{
	CUDA_SAFE_CALL(cudaFree(d_tm));
	CUDA_SAFE_CALL(cudaFree(d_endoBx));CUDA_SAFE_CALL(cudaFree(d_endoBy));CUDA_SAFE_CALL(cudaFree(d_endoBz));
	CUDA_SAFE_CALL(cudaFree(d_endoCx));CUDA_SAFE_CALL(cudaFree(d_endoCy));CUDA_SAFE_CALL(cudaFree(d_endoCz));
		 cutilSafeCall(cudaFree(d_r));
	 cutilSafeCall(cudaFree(d_rn));
	 CUDA_SAFE_CALL(cudaFree(d_tnd));


}

int main(int argc, char** argv)
{//int i;

	if(!InitCUDA()) {
		return 0;
	}

	hpc();
	printf("finish!\n");
	//for(i=0;i<3;i++) 
	// { 
	//	 //CUDA_SAFE_CALL(cudaFree(d_r[i]));CUDA_SAFE_CALL(cudaFree(d_rn[i]));
	//	CUDA_SAFE_CALL(cudaFree(d_tnd[i]))
	//};

	gpu_freetransdata();
	CUDA_SAFE_CALL(cudaFree(d_POTi));CUDA_SAFE_CALL(cudaFree(d_der));
	CUDA_SAFE_CALL(cudaFree(d_endoHnnA));CUDA_SAFE_CALL(cudaFree(d_surfPOTi));
	CUT_EXIT(argc, argv);
}
extern "C" void gpu_transdata(short int g_epicX[Nepic],short int g_epicY[Nepic],short int g_epicZ[Nepic],short int *g_tnd[3],float *g_r[3],float *g_rn[3],short int g_endoBx[NENDO*ND3],short int g_endoBy[NENDO*ND3],short int g_endoBz[NENDO*ND3],short int g_endoCx[NENDO*ND3],short int g_endoCy[NENDO*ND3],short int g_endoCz[NENDO*ND3],float g_tm[3][6])
{	//传送申请只读数据空间,并传递;申请计算用数据空间
	int i,j;
	//float *d_r[3],*d_rn[3],*d_tm;
	float cg_r[NL*3],cg_rn[NL*3];
	
	//if(!InitCUDA()) {
	//printf("CUDA error");
	//	//return 0;
	//}
	
	for(i=0;i<3;i++)
		for(j=0;j<NL;j++)
		{
		cg_r[i*NL+j]=*(g_r[i]+j);
		cg_rn[i*NL+j]=*(g_rn[i]+j);
		}
  cutilSafeCall( cudaMalloc((void**) &d_r, sizeof(float) * NL*3));
  cutilSafeCall( cudaMemcpy(d_r, cg_r, sizeof(float) * NL*3, cudaMemcpyHostToDevice));
  cutilSafeCall( cudaMalloc((void**) &d_rn, sizeof(float) * NL*3));
  cutilSafeCall( cudaMemcpy(d_rn, cg_rn, sizeof(float) * NL*3, cudaMemcpyHostToDevice));
 
  	short int cg_tnd[(NL-2)*2*3];
  	for(i=0;i<3;i++)
		for(j=0;j<(NL-2)*2;j++)
		{
		cg_tnd[i*(NL-2)*2+j]=*(g_tnd[i]+j);

		}
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_tnd, sizeof(short int) * (NL-2)*2*3));
	CUDA_SAFE_CALL( cudaMemcpy(d_tnd, cg_tnd, sizeof(short int) * (NL-2)*2*3, cudaMemcpyHostToDevice));

  //for(i=0;i<3;i++) 
	 //{
	 // //CUDA_SAFE_CALL( cudaMalloc((void**) &d_r[i], sizeof(float) * NL));
	 // //CUDA_SAFE_CALL( cudaMemcpy((d_r[i]), (g_r[i]), sizeof(float) * NL, cudaMemcpyHostToDevice));
	 // //CUDA_SAFE_CALL( cudaMalloc((void**) &d_rn[i], sizeof(float) * NL));
	 // //CUDA_SAFE_CALL( cudaMemcpy((d_rn[i]), (g_rn[i]), sizeof(float) * NL, cudaMemcpyHostToDevice));
	 // CUDA_SAFE_CALL( cudaMalloc((void**) &d_tnd[i], sizeof(short int) * (NL-2)*2));
	 // CUDA_SAFE_CALL( cudaMemcpy((d_tnd[i]), (g_tnd[i]), sizeof(short int) * (NL-2)*2, cudaMemcpyHostToDevice));
	 //};

  	float cg_tm[3*6];
	for(i=0;i<3;i++)
		for(j=0;j<6;j++)
		{
		cg_tm[i*6+j]=*(g_tm[i]+j);
		}
  	CUDA_SAFE_CALL( cudaMalloc((void**) &d_tm, sizeof(float) * 3 * 6));
	CUDA_SAFE_CALL( cudaMemcpy(d_tm, cg_tm, (sizeof(float) * 3 * 6), cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL( cudaMalloc((void**) &d_epicX, sizeof(short int) * Nepic));
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_epicY, sizeof(short int) * Nepic));
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_epicZ, sizeof(short int) * Nepic));
	CUDA_SAFE_CALL( cudaMemcpy((d_epicX),(g_epicX) , (sizeof(short int) * Nepic), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_epicY,g_epicY , sizeof(short int) * Nepic, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_epicZ, g_epicZ, sizeof(short int) * Nepic, cudaMemcpyHostToDevice));



	CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoBx, sizeof(short int) * NENDO*ND3));
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoBy, sizeof(short int) * NENDO*ND3));
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoBz, sizeof(short int) * NENDO*ND3));

	CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoCx, sizeof(short int) * NENDO*ND3));
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoCy, sizeof(short int) * NENDO*ND3));
	CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoCz, sizeof(short int) * NENDO*ND3));


	CUDA_SAFE_CALL( cudaMemcpy((d_endoBx),(g_endoBx) , (sizeof(short int) * NENDO*ND3), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_endoBy,g_endoBy , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_endoBz, g_endoBz, sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL( cudaMemcpy(d_endoCx,g_endoCx , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_endoCy,g_endoCy , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL( cudaMemcpy(d_endoCz,g_endoCz , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
//申请计算用数据空间,这样只要一次申请
  CUDA_SAFE_CALL( cudaMalloc((void**) &d_epicPOTold, sizeof(float) * Nepic));
  CUDA_SAFE_CALL( cudaMalloc((void**) &d_POTi, sizeof(float) * NL));
  CUDA_SAFE_CALL( cudaMalloc((void**) &d_der, sizeof(float) * NL));
  CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoHnnA, sizeof(float) * 2*NENDO*ND3));
  CUDA_SAFE_CALL( cudaMalloc((void**) &d_surfPOTi, sizeof(float) * (NL-2)*2));

}
//extern "C" void gpu_BSPitmm_Malloc(float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi)
//{
//  CUDA_SAFE_CALL( cudaMalloc((void**) &d_epicPOTold, sizeof(float) * Nepic));
//  CUDA_SAFE_CALL( cudaMalloc((void**) &d_POTi, sizeof(float) * NL));
//  CUDA_SAFE_CALL( cudaMalloc((void**) &d_der, sizeof(float) * NL));
//  CUDA_SAFE_CALL( cudaMalloc((void**) &d_endoHnnA, sizeof(float) * 2*NENDO*ND3));
//  CUDA_SAFE_CALL( cudaMalloc((void**) &d_surfPOTi, sizeof(float) * (NL-2)*2));
//}

extern "C" void gpu_BSPitmm_HostToDevice(float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi)
{
  cudaMemset(d_epicPOTold, 0, sizeof(float) * Nepic);
  cudaMemset(d_POTi, 0, sizeof(float) * NL);
  cudaMemset(d_der, 0,  sizeof(float) * NL);
  cudaMemset(d_endoHnnA, 0,  sizeof(float) * 2*NENDO*ND3);
  cudaMemset(d_surfPOTi, 0,  sizeof(float) * (NL-2)*2);
  //CUDA_SAFE_CALL( cudaMemcpy((d_POTi), (g_POTi), sizeof(float) * NL, cudaMemcpyHostToDevice));
  //CUDA_SAFE_CALL( cudaMemcpy((d_der), (g_der), sizeof(float) * NL, cudaMemcpyHostToDevice));
  //CUDA_SAFE_CALL( cudaMemcpy((d_endoHnnA), (g_endoHnnA), sizeof(float) * 2*NENDO*ND3, cudaMemcpyHostToDevice));
  //CUDA_SAFE_CALL( cudaMemcpy((d_surfPOTi), (g_surfPOTi), sizeof(float) * (NL-2)*2, cudaMemcpyHostToDevice));
}

extern "C" void gpu_BSPitmm_DeviceToHost(float *g_epicPOTold,float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi)
{
  CUDA_SAFE_CALL( cudaMemcpy((g_epicPOTold), (d_epicPOTold), sizeof(float) * Nepic, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL( cudaMemcpy((g_POTi), (d_POTi), sizeof(float) * NL, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL( cudaMemcpy((g_der), (d_der), sizeof(float) * NL, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL( cudaMemcpy((g_endoHnnA),(d_endoHnnA) , sizeof(float) * 2*NENDO*ND3, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL( cudaMemcpy((g_surfPOTi),(d_surfPOTi) , sizeof(float) * (NL-2)*2, cudaMemcpyDeviceToHost));
}

extern "C" void gpu_dpl_all(short int do_epicPOT,float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL],
							float g_HRTx0,float g_HRTy0,float g_HRTz0,int g_NendoB,int g_NendoC,
						float *g_endoHnnA,short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6],float *g_epicPOTold)
{
	float * d_dpl;
	cutilSafeCall( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	cutilSafeCall( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

	  k_dpl_nPos<<<1, g_nPos>>>(g_posi,g_posj,g_posk,g_nPos,d_dpl,d_POTi,d_der,d_r ,d_rn);
	//if (g_offset<100)
	//{
		k_dpl_Nendo<<<6, 512>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoB,0,d_dpl,d_endoHnnA,d_endoBx,d_endoBy,d_endoBz,d_tm);
	//}
	//else
	//{	
		k_dpl_Nendo<<<6, 512>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoC,g_NendoB,d_dpl,d_endoHnnA,d_endoCx,d_endoCy,d_endoCz,d_tm);
	//};
	
	k_dpl_nPos_2<<<2, 342>>>(g_posi,g_posj,g_posk,d_dpl,d_r,d_surfPOTi,d_tnd);

	if (do_epicPOT==1) k_dpl_Nepic<<<Nepic/512+1, 512>>>(d_epicX,d_epicY,d_epicZ,g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,d_dpl,d_epicPOTold,d_tm,Nepic);

	cutilSafeCall(cudaFree(d_dpl));


}

extern "C" void gpu_dpl_Nepic(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
							  float g_dpl[3],float g_tm[3][6],float *g_epicPOTold)
{
	float * d_dpl;
	cutilSafeCall( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	cutilSafeCall( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));
	k_dpl_Nepic<<<Nepic/512+1, 512>>>(d_epicX,d_epicY,d_epicZ,g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,d_dpl,d_epicPOTold,d_tm,Nepic);
cutilSafeCall(cudaFree(d_dpl));
}

extern "C" void gpu_dpl_Nendo(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
							  int g_NendoBC,int g_offset,float g_dpl[3],float *g_endoHnnA,
							  short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6])
{
 //k_dpl_Nendo<<<1, g_NendoBC>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,g_offset,g_dpl,d_endoHnnA,d_endoBx,d_endoBy,d_endoBz,d_tm);
 //   numberofb=g_NendoBC;
	//while(g_NendoBC!=0)
	float * d_dpl;
	cutilSafeCall( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	cutilSafeCall( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

	if (g_offset<100)
	{
		k_dpl_Nendo<<<6, 512>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,g_offset,d_dpl,d_endoHnnA,d_endoBx,d_endoBy,d_endoBz,d_tm);
	}
	else
	{	k_dpl_Nendo<<<6, 512>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,g_offset,d_dpl,d_endoHnnA,d_endoCx,d_endoCy,d_endoCz,d_tm);
	};
	
	cutilSafeCall(cudaFree(d_dpl));
	//k_dpl_Nendo<<<1, (g_NendoBC-512*5)>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,(g_offset+512*5),g_dpl,d_endoHnnA,d_endoBx,d_endoBy,d_endoBz,d_tm);

}
extern "C" void gpu_dpl_nPos_2(float g_posi,float g_posj,float g_posk,float g_dpl[3])
{
		float * d_dpl;
	cutilSafeCall( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	cutilSafeCall( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

k_dpl_nPos_2<<<2, 342>>>(g_posi,g_posj,g_posk,d_dpl,d_r,d_surfPOTi,d_tnd);
	  cutilSafeCall(cudaFree(d_dpl));

}
extern "C" void gpu_dpl_nPos(float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL])
{
	float * d_dpl;
	cutilSafeCall( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	cutilSafeCall( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

	//float *d_POTi=0, *d_der=0;
	//  CUDA_SAFE_CALL( cudaMalloc((void**) &d_POTi, sizeof(float) * NL));
	//  CUDA_SAFE_CALL( cudaMalloc((void**) &d_der, sizeof(float) * NL));
	//  CUDA_SAFE_CALL( cudaMemcpy((d_POTi), (g_POTi), sizeof(float) * NL, cudaMemcpyHostToDevice));
	//  CUDA_SAFE_CALL( cudaMemcpy((d_der), (g_der), sizeof(float) * NL, cudaMemcpyHostToDevice));

	  k_dpl_nPos<<<1, g_nPos>>>(g_posi,g_posj,g_posk,g_nPos,d_dpl,d_POTi,d_der,d_r ,d_rn);
	  
	  cutilSafeCall(cudaFree(d_dpl));

	//k_dpl_nPos<<<1, g_nPos>>>(g_posi,g_posj,g_posk,g_nPos,g_dpl,d_POTi,d_der,d_r,d_rn);

	  //CUDA_SAFE_CALL( cudaMemcpy((g_POTi), (d_POTi), sizeof(float) * NL, cudaMemcpyDeviceToHost));
	  //CUDA_SAFE_CALL( cudaMemcpy((g_der), (d_der), sizeof(float) * NL, cudaMemcpyDeviceToHost));		
	  //CUDA_SAFE_CALL(cudaFree(d_der));
	  //CUDA_SAFE_CALL(cudaFree(d_POTi));
//extern "C" void dplpro(float *POTi,const short int NL, const float **r)

//	float *d_data=0,*d_r[3],;
//	printf("%f,%f\n", *POTi,*(POTi+1));
//	for(int i=0;i<3,i++) CUDA_SAFE_CALL( cudaMalloc((void**) &d_data, sizeof(float) * NL*4));
//	CUDA_SAFE_CALL( cudaMalloc((void**) &d_data, sizeof(float) * NL*4));
//	CUDA_SAFE_CALL( cudaMemcpy(d_data,POTi , sizeof(float) * NL*4, cudaMemcpyHostToDevice));
//		dpl<<<1, 16>>>(d_data);
//		CUDA_SAFE_CALL( cudaMemcpy(POTi, d_data, sizeof(float) * NL*4, cudaMemcpyDeviceToHost));
//	printf("%f,%f\n", *POTi,*(POTi+1));
//
//
}


/************************************************************************/
/* Example                                                              */
/************************************************************************/
//__global__ static void HelloCUDA(char* result, int num)
//{
//	int i = 0;
//	char p_HelloCUDA[] = "Hello CUDA!";
//	for(i = 0; i < num; i++) {
//		result[i] = p_HelloCUDA[i];
//	}
//}

/************************************************************************/
/* HelloCUDA                                                            */
/************************************************************************/
//extern "C" void test(const int argc, const char** argv)
//{
//	if(!InitCUDA()) {
//		return;
//	}
//
//	char	*device_result	= 0;
//	char	host_result[12]	={0};
//
//	CUDA_SAFE_CALL( cudaMalloc((void**) &device_result, sizeof(char) * 11));
//
//	unsigned int timer = 0;
//	CUT_SAFE_CALL( cutCreateTimer( &timer));
//	CUT_SAFE_CALL( cutStartTimer( timer));
//
//	HelloCUDA<<<1, 1, 0>>>(device_result, 11);
//	CUT_CHECK_ERROR("Kernel execution failed\n");
//
//	CUDA_SAFE_CALL( cudaThreadSynchronize() );
//	CUT_SAFE_CALL( cutStopTimer( timer));
//	printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
//	CUT_SAFE_CALL( cutDeleteTimer( timer));
//
//	CUDA_SAFE_CALL( cudaMemcpy(&host_result, device_result, sizeof(char) * 11, cudaMemcpyDeviceToHost));
//	printf("%s\n", host_result);
//
//	CUDA_SAFE_CALL( cudaFree(device_result));
//	CUT_EXIT(argc, argv);
//
//	return;
//}
/*extern "C" void gpu_dpl_all(short int do_epicPOT,float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL],
							float g_HRTx0,float g_HRTy0,float g_HRTz0,int g_NendoB,int g_NendoC,
						float *g_endoHnnA,short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6],float *g_epicPOTold);

extern "C" void gpu_dpl_nPos(float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL]);
extern "C" void gpu_dpl_nPos_2(float g_posi,float g_posj,float g_posk,float g_dpl[3]);
extern "C" void gpu_dpl_Nendo(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
							  int g_NendoBC,int g_offset,float g_dpl[3],float *g_endoHnnA,
							  short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6]);
extern "C" void gpu_dpl_Nepic(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
							  float g_dpl[3],float g_tm[3][6],float *g_epicPOTold);
__global__ void k_dpl_nPos(float k_posi,float k_posj,float k_posk,int k_nPos,float *k_dpl,
								  float *k_POTi,float *k_der,float *k_r ,float *k_rn )
__global__ static void k_dpl_nPos_2(float k_posi,float k_posj,float k_posk,float *k_dpl,float *k_r,float *d_surfPOTi,
									short int *d_tnd)
__global__ static void k_dpl_Nendo(float k_posi,float k_posj,float k_posk,
								   float k_HRTx0,float k_HRTy0,float k_HRTz0,int k_NendoB,int k_offset,float *k_dpl,
								   float *k_endoHnnA,short int *k_endoBx,short int *k_endoBy,short int *k_endoBz,
								   float *k_tm)
__global__ static void k_dpl_Nepic(short int *k_epicX,short int *k_epicY,short int *k_epicZ,float k_posi,float k_posj,float k_posk,
								   float k_HRTx0,float k_HRTy0,float k_HRTz0,float *k_dpl,float *k_epicPOTold,
								   float *k_tm,short int k_Nepic)							   
__global__ static void k_dpl_all(short int *k_epicX,short int *k_epicY,short int *k_epicZ,float k_posi,float k_posj,float k_posk,int k_nPos,float *k_POTi,
								   float *k_der,float *k_r ,float *k_rn,float *d_surfPOTi,short int *d_tnd,float k_HRTx0,float k_HRTy0,float k_HRTz0,
								   int k_NendoB,int k_offset,float *k_dpl,float *k_epicPOTold,float *k_endoHnnA,short int *k_endoBx,short int *k_endoBy,
								   short int *k_endoBz,float *k_tm,short int k_Nepic)								   
{
float ax,ay,az,r1,r2,r3,dr,rv3,tmp1,tmp2,tmp3,r5,ds,bx,by,bz,ret_der,ret_POTi;
int n0,n1,n2;
int n=blockDim.x * blockIdx.x + threadIdx.x;
	//int n=threadIdx.x;nPOS
if (n< k_Nepic)			
	{					//for (n=0; n<Nepic; ++n) {
						//ax=HRTx0+epicX[n]*tmswf[0][0]+epicY[n]*tmswf[0][1]+epicZ[n]*tmswf[0][2]-posi;
						//ay=HRTy0+epicX[n]*tmswf[1][0]+epicY[n]*tmswf[1][1]+epicZ[n]*tmswf[1][2]-posj;
						//az=HRTz0+epicX[n]*tmswf[2][0]+epicY[n]*tmswf[2][1]+epicZ[n]*tmswf[2][2]-posk;
		ax=k_HRTx0;
		tmp1=*(k_epicX+n) * *(k_tm);
		ax=ax+tmp1;
		tmp2=*(k_epicY+n) * *(k_tm+1);
		ax=ax+tmp2;
		tmp3=*(k_epicZ+n) * *(k_tm+2);
		ax=ax+tmp3;
		ax=ax-k_posi;//ax
		ay=k_HRTy0;
		tmp1=*(k_epicX+n) * *(k_tm+1*6);
		ay=ay+tmp1;
		tmp2=*(k_epicY+n) * *(k_tm+1*6+1);
		ay=ay+tmp2;
		tmp3=*(k_epicZ+n) * *(k_tm+1*6+2);
		ay=ay+tmp3;
		ay=ay-k_posj;//ay
		az=k_HRTz0;
		tmp1=*(k_epicX+n) * *(k_tm+2*6);
		az=az+tmp1;
		tmp2=*(k_epicY+n) * *(k_tm+2*6+1);
		az=az+tmp2;
		tmp3=*(k_epicZ+n) * *(k_tm+2*6+2);
		az=az+tmp3;
		az=az-k_posk;//az

		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		//dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
		tmp1=k_dpl[0]*ax;
		dr=tmp1;
		tmp2=k_dpl[1]*ay;
		dr+=tmp2;
		tmp3=k_dpl[2]*az;
		dr+=tmp3;//dr

		rv3=1/r3;
		*(k_epicPOTold+n)+=dr*rv3;
	}
	
	if (n< k_NendoB)			
	{
		//ax=k_HRTx0+*(k_endoBx+n) * *(k_tm)+*(k_endoBy+n) * *(k_tm+1)+*(k_endoBz+n) * *(k_tm+2)-k_posi;
		//ay=k_HRTy0+*(k_endoBx+n) * *(k_tm+1*6)+*(k_endoBy+n) * *(k_tm+1*6+1)+*(k_endoBz+n) * *(k_tm+1*6+2)-k_posj;
		//az=k_HRTz0+*(k_endoBx+n) * *(k_tm+2*6)+*(k_endoBy+n) * *(k_tm+2*6+1)+*(k_endoBz+n) * *(k_tm+2*6+2)-k_posk;
		ax=k_HRTx0;
		tmp1=*(k_endoBx+n) * *(k_tm);
		ax=ax+tmp1;
		tmp2=*(k_endoBy+n) * *(k_tm+1);
		ax=ax+tmp2;
		tmp3=*(k_endoBz+n) * *(k_tm+2);
		ax=ax+tmp3;
		ax=ax-k_posi;//ax
		ay=k_HRTy0;
		tmp1=*(k_endoBx+n) * *(k_tm+1*6);
		ay=ay+tmp1;
		tmp2=*(k_endoBy+n) * *(k_tm+1*6+1);
		ay=ay+tmp2;
		tmp3=*(k_endoBz+n) * *(k_tm+1*6+2);
		ay=ay+tmp3;
		ay=ay-k_posj;//ay
		az=k_HRTz0;
		tmp1=*(k_endoBx+n) * *(k_tm+2*6);
		az=az+tmp1;
		tmp2=*(k_endoBy+n) * *(k_tm+2*6+1);
		az=az+tmp2;
		tmp3=*(k_endoBz+n) * *(k_tm+2*6+2);
		az=az+tmp3;
		az=az-k_posk;//az


		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		//dr=k_dpl[0]*ax+k_dpl[1]*ay+k_dpl[2]*az;
		tmp1=k_dpl[0]*ax;
		dr=tmp1;
		tmp2=k_dpl[1]*ay;
		dr+=tmp2;
		tmp3=k_dpl[2]*az;
		dr+=tmp3;//dr

		rv3=1/r3;
		*(k_endoHnnA+k_offset+n)+=dr*rv3;
	}
	
	if (n< ((NL-2)*2))			
	{
		n0=d_tnd[n]-1;
		n1=d_tnd[(NL-2)*2+n]-1;
		n2=d_tnd[(NL-2)*2*2+n]-1;
		ax=(k_r[n0]+k_r[n1]+k_r[n2])/3-k_posi;
		ay=(k_r[NL+n0]+k_r[NL+n1]+k_r[NL+n2])/3-k_posj;
		az=(k_r[2*NL+n0]+k_r[2*NL+n1]+k_r[2*NL+n2])/3-k_posk;
		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		dr=ax;
		dr=dr*k_dpl[0];
		dr+=k_dpl[1]*ay;
		dr+=k_dpl[2]*az;
		rv3=1/r3;
		*(d_surfPOTi+n)+=dr*rv3;

	}
	if (n< nPos)
	{			
	ax=k_r[n];
	ay=k_r[NL+n];
	az=k_r[2*NL+n];
	ax = ax - k_posi;
	ay = ay - k_posj;
	az = az - k_posk;

	r2=ax*ax+ay*ay+az*az;
	r1=(float)sqrt(r2);
	r3=(float)(r1*r2);
	r5=(float)(r2*r3);
	dr=k_dpl[0]*ax+k_dpl[1]*ay+k_dpl[2]*az;
	ds=3*dr/r5;
	rv3=1/r3;
	bx=k_dpl[0]*rv3-ax*ds;
	by=k_dpl[1]*rv3-ay*ds;
	bz=k_dpl[2]*rv3-az*ds;
		//*(k_der+n)+=*(d_rn[0]+n)*bx+*(d_rn[1]+n)*by+*(d_rn[2]+n)*bz;
	ret_der  = k_der[n];
	ret_der += k_rn[n]*bx;
	ret_der += k_rn[NL+n]*by;
	ret_der += k_rn[2*NL+n]*bz;
	k_der[n] = ret_der;

	//*(k_POTi+n)+=dr*rv3;
	ret_POTi = k_POTi[n];
	ret_POTi += dr*rv3;
	k_POTi[n] = ret_POTi;
	 //__syncthreads();
	}
	
}
*/