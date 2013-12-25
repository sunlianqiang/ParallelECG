#include <omp.h>
#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <crtdbg.h>
#include <cuda_runtime.h>
#include <afxwin.h>
#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#define CRTDBG_MAP_ALLOC

const short int ND = 1;
const short int ND3 = 1;
const short int NI = 56;
const short int NJ = 56;
const short int NK = 90;
const short int NL = 344;
const short int NPARM = 35;
const short int NCELL = 14;
const short int INFTIME = 9999;
const short int ANISO = 1; // aniso switch
const short int NCYCL = 20; // max cycle num
const short int TSTEP = 2000;
const short int NENDO = 4000;

char flg_thread=1,flg_display,flg_calculate,flg_stop;
char flg_calcu_option;
float HRTscale,HRTx0,HRTy0,HRTz0,phai,pusai,theta;
short int ipttl[4][NI*ND*ND],nttl,idist,iHB[4][50*ND3],iBB[3][50*ND3];
short int kTop,kBtm,kVtr,nHB,nBB;

short int ic,ibbDLY,idltt;
short int ibbSTEP;
char flag_flop;
char *mapCell[NK];
short int nPos,nv[3],maxXctStep;
short int *iparm;
float *ydata[NCELL];
float tmswf[3][6],alp;
short int la012[NCELL],la0123[NCELL];
float *r[3],*rn[3];
float *aw[NL], *bw;
short int *kmin, *kmax;
short int *mag[4];
short int *mapAPD[NK];
short int *mapACT[NK];
short int   nbbSTEP;

short int *mapSpeed[NK];

short int *mapXCTm[NCYCL];  // store the exciting time 保存兴奋时间
int NendoB, NendoC;
short int endoBx[NENDO*ND3];
short int endoBy[NENDO*ND3];
short int endoBz[NENDO*ND3];
short int endoCx[NENDO*ND3];
short int endoCy[NENDO*ND3];
short int endoCz[NENDO*ND3];

// epicardial variable 心外膜变量
const short int Nepic=NI*NJ*2;//short int Nepic 心外膜计算个数
vector<short int> epicX;  // x
vector<short int> epicY;  // y
vector<short int> epicZ;  // z
short int epicX_old[Nepic];
short int epicY_old[Nepic];
short int epicZ_old[Nepic];

float *POTi;
float *POT[NL],*POT_reduce[NL];//*POT_reduce[NL] by sf 090622
float VCG[3],bufVCG[2][3],bufGRD;
short int nTimeStep,itbuf,nextStep;
short int *iStep;
char answer;

long mNub;
long *locXCT[NK];
long totalCell;
// anisotropy variables 各个方向的变量
short int maxlay;
float *fibdir[3];
float vl2[10], vt2[10], rrat1;
float planedir[3][30];
float prx[12][12], pry[12][12], prz[12][12];
float xaxis[3],yaxis[3],zaxis[3];

short int mBCL,miBN,mxcycle,idltc,mS2ST,mS2CL,mS2BN;
short int ipstm[3][NI*ND*ND];
short int vHB[NCYCL][50*ND3];
short int  excited=0;

CString dataPath="E:\\chuan100\\";

const short int useGPU=1,gpuspeed=17;//by sf 090403 useCPU 1--yes 0--no gpuspeed 1 or 17
short int GPUnum=1,corenum=0;//by sf 090823 the number of GPU device,allnum按GPUnum,corenum次序存 
// 线程数量
short int threadnum=4;//by sf 090403 threadnum<0 auto >0 set number of thread=threadnum
short int iTimebegin=1,iTimeend;

float **gatheralldpl;//by sf 090408  for write dpl[3] in BSPitmm
int **gatherallijk,*countallijk,*countallijk_reduce,*itask[2],*iloops[3],isumdipoles=0;//,*iTimetid;//by sf 090408  for write the ijk of dpl[3] in BSPitmm
double starttime,endtime; 
double bsptime[4] = {0.0,0.0,0.0,0.0};
int BSPitmmcount(short int iTime0);

void rdHRT(void);
void rdpos(void);
void rdnod(void);
void rdmtx(void);
void rdelc(void);
void locfile(void);
void ECGcal(void);
void geoinfc(void);
void setaniso(void);
void neibdir(void);
void stminvx(short int);
void XCTinvcm(void);
void fibplane(void);
void fibdirct(void);
void savACT(void);
//void savACT(int myid);
void freeFibdir(void);
void freemapAPDcs(void);
void freemapAPD(void);
void freebrs(void);
void freemagcs(void);
void freePOTcs(void);

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

//------------ 2009-2-6-16 BY SWF---------
// comment:
extern "C" short int cudamain(int argc, char** argv);
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
//------------ 2009-2-6-16 BY SWF---------



//int main(int argc,char *argv[])
//void hpc(int argc, char** argv)
void main(int argc, char** argv)
{	
	int  myid, numprocs;
	int  namelen;

	//------------ 2009-2-6-16 BY SWF---------
	// comment:


	FILE *fptime;

	//------------ 2009-2-6-16 BY SWF---------
	short int ipttl[4][56];
	HFILE hFp;
	short int nVCG,BSPm,mTime,iTime,i,j,k;
	short int nsnrt;
	float *VCGs[3];
	float eff;
	float *endoHnnA;
	float *endoPOT[TSTEP];

	short int index;
	int nn,n0,n1,n2,ni;
	float pi=3.14159;
	short int *tnd[3];

	int li;
	void XCTcalm(void);
	//void XCTcalm(int myid);
	void BSPcalm(void);
	void rdAPDm(void);
	void freeXCTm(void);

	fprintf(stdout, "Begin computing. %f\n", clock());

	for(i=0;i<NK;i++) {
		mapCell[i] = (char *) malloc(NI*NJ);
		mapAPD[i] = (short int *) malloc(NI*NJ*2);
		mapSpeed[i] = (short int *) malloc(NI*NJ*2);
		mapACT[i] = (short int *) malloc(NI*NJ*2);
		locXCT[i] = (long *) malloc(NI*NJ*4);
		if((mapCell[i]==NULL)||(mapAPD[i]==NULL)||(mapACT[i]==NULL)||(locXCT[i]==NULL)) {
			fprintf(stdout,"out of memory\n");
			fflush(stdout);
			return;		
		}
	}  
	iparm = (short int *) malloc(NCELL*NPARM*2);
	kmin = (short int *) malloc(NI*NJ*2);
	kmax = (short int *) malloc(NI*NJ*2);
	iStep = (short int *) malloc(TSTEP*2);
	if((iparm==NULL)||(kmin==NULL)||(kmax==NULL)||(iStep==NULL)) {
		fprintf(stdout,"out of memory\n");
		fflush(stdout);
		return;
	}
	if(flg_thread==0) {
		freemapAPDcs();
		freemapAPD();
		return;
	}
	for(i=0;i<3;i++) {
		r[i] = (float *) malloc(NL*4);
		rn[i] = (float *) malloc(NL*4);
		if((r[i]==NULL)||(rn[i]==NULL)) {
			fprintf(stdout,"out of memory\n");
			fflush(stdout);
			return;
		}
	}    
	for(i=0;i<NCELL;i++) {
		ydata[i] = (float *) malloc(1000*ND*4);
		if(ydata[i]==NULL) {
			fprintf(stdout,"out of memory\n");
			fflush(stdout);
			return;		}
	}  
	if(flg_thread==0) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		return;
	}
	for(i=0;i<4;i++) {
		mag[i] = (short int *) malloc(50000*ND3*2);
		if(mag[i]==NULL) {
			fprintf(stdout,"out of memory\n");
			fflush(stdout);
			return;		}
	}   
	for(k=0;k<NK;k++) { 
		for(j=0;j<NJ;j++) {
			for(i=0;i<NI;i++) {
				*(mapAPD[k]+j*NI+i)=0;
				*(mapSpeed[k]+j*NI+i)=0;
			}
		}
	}

	for(i=0;i<4;i++) {
		for(li=0;li<50000*ND3;li++) {
			*(mag[i]+li)=0;
		}
	}
	//TRACE("\nReading HRT file ...");	
	rdHRT();

	if(flag_flop||(flg_thread==0)) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		return;
	}
	//TRACE("\nReading APD file ...");
	rdAPDm();
	if(flag_flop||(flg_thread==0)) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		return;
	}
	//TRACE("\nReading POS file ...");
	rdpos();
	if(flag_flop||(flg_thread==0)) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		return;
	}
	//TRACE("\nReading NOD file ...");
	rdnod();
	if(flag_flop||(flg_thread==0)) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		return;
	}
	//TRACE("\nReading ELC file ...");

	rdelc();
	if(flag_flop||(flg_thread==0)) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		return;
	}
	//TRACE("\nLocating Cell Sequence ...");

	locfile();
	if(flg_thread==0) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		return;
	}
	//TRACE("\nFinding Geometric Info ...");

	geoinfc(); 
	if(flg_thread==0) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		return;
	}

	if (ANISO==1) {
		//TRACE("\nCalculating Rotating Anisotropy ...");
		for (i=0; i<3; i++) {
			fibdir[i] = (float *) malloc(50000*ND3*4);
			if (fibdir[i]==NULL) {
				fprintf(stdout,"Out of memory ! !\n");
				fflush(stdout);
				return;// 0;
			}
		}   

		for(i=0;i<3;i++) {
			for(li=0;li<50000*ND3;li++) {
				*(fibdir[i]+li)=0.;
			}
		}
		//TRACE("\nCalculating setaniso ...");
		setaniso();
		if(flg_thread==0) {
			freemapAPDcs();
			freemapAPD();
			freebrs();
			freemagcs();
			freeFibdir();
			return;
		}
		//TRACE("\nCalculating neibdir ...");
		neibdir();
		if(flg_thread==0) {
			freemapAPDcs();
			freemapAPD();
			freebrs();
			freemagcs();
			freeFibdir();
			return;
		}
		//TRACE("\nCalculating stminvx ...");
		stminvx(50);
		if(flg_thread==0) {
			freemapAPDcs();
			freemapAPD();
			freebrs();
			freemagcs();
			freeFibdir();
			return;
		}
		//TRACE("\nCalculating XCTinvcm ...");
		XCTinvcm();
		if(flg_thread==0) {
			freemapAPDcs();
			freemapAPD();
			freebrs();
			freemagcs();
			freeFibdir();
			return;
		}
		//TRACE("\nCalculating fibplane ...");
		fibplane();
		if(flg_thread==0) {
			freemapAPDcs();
			freemapAPD();
			freebrs();
			freemagcs();
			freeFibdir();
			return;
		}
		//TRACE("\nCalculating fibdirct ...");
		fibdirct();
		if(flg_thread==0) {
			freemapAPDcs();
			freemapAPD();
			freebrs();
			freemagcs();
			freeFibdir();
			return;
		}
		//TRACE("\nCompleting Rotating Anisotropy ...");	
	}

	//TRACE("\nStimulus calculating ...");
	stminvx(20*ND);
	if(flg_thread==0) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freemagcs();
		freeFibdir();
		return;
	}
	//TRACE("\nExcitation estimating ...");
	XCTinvcm(); 

	savACT();
	fflush(stdout);

	freemagcs();

	for(i=0;i<NCYCL;i++) {
		mapXCTm[i]=(short int *) malloc(50000*ND3*2);
		if((mapXCTm[i]==NULL)) {
			fprintf(stdout,"Out of memory ! !\n");
			fflush(stdout);
			return;// 0; 
		}
	}  
	if(flg_thread==0) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freeXCTm();
		freeFibdir();
		return;
	}

	//TRACE("\nExcitation calculating ...");
	XCTcalm();
	//fprintf(stdout,"XCTcalm()ok=;myid=%d \n",myid);
	fflush(stdout);
	if(flg_thread==0) {
		freemapAPDcs();
		freemapAPD();
		freebrs();
		freeXCTm();
		freeFibdir();
		return;
	}

	if(!flg_calcu_option) {
		for(i=0;i<NL;i++) {
			POT[i]=(float *) malloc(TSTEP*4);
			POT_reduce[i]=(float *) malloc(TSTEP*4);//by sf 090622
			aw[i]=(float *) malloc(NL*4);
			if((POT[i]==NULL)||(aw[i]==NULL)) {
				// MessageBox(NULL,"Out of memory !",NULL,MB_OK);
				cout<<"Out of memory !"<<endl;
				exit(0);
			}
		}
		for(i=0;i<NL;i++) {
			for(j=0;j<TSTEP;j++) {
				*(POT[i]+j)=(float)0;
				*(POT_reduce[i]+j)=(float)0;//by sf 090622
			}
		}		

		bw=(float *) malloc(NL*4);
		POTi=(float *) malloc(NL*4);
		if((POTi==NULL)||(bw==NULL)) {
			// MessageBox(NULL,"Out of memory !",NULL,MB_OK);
			cout<<"Out of memory !"<<endl;
			exit(0);
		}  
		for(i=0;i<NL;i++) *(POTi+i)=(float)0;
		if(flg_thread==0) {
			freemapAPD();
			freemapAPDcs();
			freebrs();
			freeXCTm();
			freePOTcs();
			freeFibdir();
			return;
		}
		//TRACE("\nReading MTX file ...");		
		rdmtx();
		if(flag_flop||(flg_thread==0)) {
			freemapAPD();
			freemapAPDcs();
			freebrs();
			freeXCTm();
			freePOTcs();
			freeFibdir();
			return;
		}
		//TRACE("\nBSPM calculating ...");
		//------------ 2009-2-4-15 BY SWF---------
		// comment:  test data trans
		//int mydata[2]={20,60};
		//printf("my%d,%d\n", mydata[0],mydata[1]);
		//printf("aa%f,%f\n", *POTi,*(POTi+1));
		//dplpro(POTi,NL,r);
		//printf("aa%f,%f\n", *POTi,*(POTi+1));
		//printf("my%d,%d\n", mydata[0],mydata[1]);
		//------------ 2009-2-4-15 BY SWF---------

		//------------ 2009-2-6-16 BY SWF---------
		// comment:
		starttime = clock();
		fprintf(stdout,"starttime = %f\n", starttime);

		//------------ 2009-2-6-16 BY SWF---------
		if (useGPU==1) 
		{			
			GPUnum=cudamain(argc, argv);		
			fprintf(stdout,"GPUnum = %d", GPUnum);
		};
		BSPcalm();

		//------------ 2009-2-6-16 BY SWF---------
		// comment:
		endtime = clock();
		fprintf(stdout,"\nendtime = %f\n", endtime);
		/*if (myid==0)
		{
			//fprintf(stdout,"sd test- endtime = %f,all-time = %f,threadnum=%d,useGPU=%d,numprocs=%d,nTimeStep=%d\n", starttime,(endtime-starttime)/CLK_TCK,threadnum,useGPU,numprocs,nTimeStep);
			fprintf(stdout,"sd test all-time=%f,useGPU=%d,threadnum=%d,numprocs=%d,nTimeStep=%d\n",(endtime-starttime)/CLK_TCK,useGPU,threadnum,numprocs,nTimeStep);
			fptime=fopen(dataPath+"gputime.txt","a")  ;
			fprintf(fptime,"sd test all-time=%f,useGPU=%d,threadnum=%d,numprocs=%d,nTimeStep=%d\n",(endtime-starttime)/CLK_TCK,useGPU,threadnum,numprocs,nTimeStep);
			fclose(fptime);
			//fptime=fopen(dataPath+"task.txt","a")  ;
			//fprintf(fptime,"sd test all-time=%f,useGPU=%d,threadnum=%d,numprocs=%d,nTimeStep=%d\n",(endtime-starttime)/CLK_TCK,useGPU,threadnum,numprocs,nTimeStep);
			//for(i=0;i<2;i=i+1)
			//{
			//	for(j=0;j<=nTimeStep;j=j+1)
			//	{
			//		fprintf(fptime,"itask[%d][%d]=%d\n",i,j,*(itask[i]+j));
			//	}
			//};
			//for(i=0;i<3;i=i+1)
			//{
			//	for(j=0;j<=nTimeStep;j=j+1)
			//	{
			//		fprintf(fptime,"iloops[%d][%d]=%d\n",i,j,*(iloops[i]+j));
			//	}
			//};
			//fclose(fptime);
		}*/
		//------------ 2009-2-6-16 BY SWF---------


		if(flag_flop||(flg_thread==0)) {
			freemapAPD();
			freemapAPDcs();
			freebrs();
			freeXCTm();
			freePOTcs();
			freeFibdir();
			return;
		}
		//TRACE("\nECG and VCG calculating ...");

		ECGcal();
	}
	freemapAPD();
	freemapAPDcs();
	freebrs();
	freeXCTm();
	freeFibdir();
	if(!flg_calcu_option) { 
		freePOTcs();
	}

	fprintf(stdout,"Simulation End !\n");
	fflush(stdout);
	flg_thread=0;
	flg_display=0;
	flg_calculate=0;
	flg_stop=0;
	return;
}

void rdHRT(void) {
	HFILE hFp;
	short int i, j, k, nCell, index;

	hFp=_lopen(dataPath+"tour.hrt ",OF_READ);  

	if (hFp==HFILE_ERROR) {
		fprintf(stdout,"Can not open nod file ! !\n");
		fflush(stdout);
		flag_flop=1;
		return;
	}

	_lread(hFp,&nttl,2);

	if (nttl>NI/ND) nttl=NI/ND;

	/**
	* read stimulation cell's position
	*/
	for (i=0;i<nttl;i++) {
		_lread(hFp,&ipttl[0][i*ND3],2); 
		_lread(hFp,&ipttl[1][i*ND3],2); 
		_lread(hFp,&ipttl[2][i*ND3],2);
		_lread(hFp,&ipttl[3][i*ND3],2);
	}

	if (ND == 2) {
		for (i=0;i<nttl;i++) {	
			ipttl[0][i*ND3] *= ND;
			ipttl[1][i*ND3] *= ND;
			ipttl[2][i*ND3] *= ND;
			ipttl[3][i*ND3] *= ND;

			for (j = 1; j < ND3; j++) {
				ipttl[0][i*ND3+j] = ipttl[0][i*ND3]; 
				ipttl[1][i*ND3+j] = ipttl[1][i*ND3]; 
				ipttl[2][i*ND3+j] = ipttl[2][i*ND3]; 
				ipttl[3][i*ND3+j] = ipttl[3][i*ND3]; 
			}
			ipttl[0][i*ND3+1] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+2] = ipttl[1][i*ND3]+1; 

			ipttl[0][i*ND3+3] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+3] = ipttl[1][i*ND3]+1; 

			ipttl[2][i*ND3+4] = ipttl[2][i*ND3]+1; 

			ipttl[0][i*ND3+5] = ipttl[0][i*ND3]+1; 
			ipttl[2][i*ND3+5] = ipttl[2][i*ND3]+1; 

			ipttl[1][i*ND3+6] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+6] = ipttl[2][i*ND3]+1; 

			ipttl[0][i*ND3+7] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+7] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+7] = ipttl[2][i*ND3]+1; 		
		}
	}

	if (ND == 3) {
		for (i=0;i<nttl;i++) {	
			ipttl[0][i*ND3] *= ND;
			ipttl[1][i*ND3] *= ND;
			ipttl[2][i*ND3] *= ND;
			//ipttl[3][i*ND3] *= ND;

			for (j = 1; j < ND3; j++) {
				ipttl[0][i*ND3+j] = ipttl[0][i*ND3]; 
				ipttl[1][i*ND3+j] = ipttl[1][i*ND3]; 
				ipttl[2][i*ND3+j] = ipttl[2][i*ND3]; 
				ipttl[3][i*ND3+j] = ipttl[3][i*ND3]; 
			}

			// 00(1,2)
			ipttl[2][i*ND3+1] = ipttl[2][i*ND3]+1; 
			ipttl[2][i*ND3+2] = ipttl[2][i*ND3]+2; 

			// 01(0,1,2)
			ipttl[1][i*ND3+3] = ipttl[1][i*ND3]+1; 
			ipttl[1][i*ND3+4] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+4] = ipttl[2][i*ND3]+1; 
			ipttl[1][i*ND3+5] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+5] = ipttl[2][i*ND3]+2; 

			// 02(0,1,2)
			ipttl[1][i*ND3+6] = ipttl[1][i*ND3]+2; 
			ipttl[1][i*ND3+7] = ipttl[1][i*ND3]+2; 
			ipttl[2][i*ND3+7] = ipttl[2][i*ND3]+1; 
			ipttl[1][i*ND3+8] = ipttl[1][i*ND3]+2; 
			ipttl[2][i*ND3+8] = ipttl[2][i*ND3]+2; 

			// 10(0,1,2)
			ipttl[0][i*ND3+9] = ipttl[0][i*ND3]+1; 
			ipttl[0][i*ND3+10] = ipttl[0][i*ND3]+1; 
			ipttl[2][i*ND3+10] = ipttl[2][i*ND3]+1; 
			ipttl[0][i*ND3+11] = ipttl[0][i*ND3]+1; 
			ipttl[2][i*ND3+11] = ipttl[2][i*ND3]+2; 

			// 11(0,1,2)
			ipttl[0][i*ND3+12] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+12] = ipttl[1][i*ND3]+1; 
			ipttl[0][i*ND3+13] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+13] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+13] = ipttl[2][i*ND3]+1; 
			ipttl[0][i*ND3+14] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+14] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+14] = ipttl[2][i*ND3]+2; 

			// 12(0,1,2)
			ipttl[0][i*ND3+15] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+15] = ipttl[1][i*ND3]+2; 
			ipttl[0][i*ND3+16] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+16] = ipttl[1][i*ND3]+2; 
			ipttl[2][i*ND3+16] = ipttl[2][i*ND3]+1; 
			ipttl[0][i*ND3+17] = ipttl[0][i*ND3]+1; 
			ipttl[1][i*ND3+17] = ipttl[1][i*ND3]+2; 
			ipttl[2][i*ND3+17] = ipttl[2][i*ND3]+2; 

			// 20(0,1,2)
			ipttl[0][i*ND3+18] = ipttl[0][i*ND3]+2; 
			ipttl[0][i*ND3+19] = ipttl[0][i*ND3]+2; 
			ipttl[2][i*ND3+19] = ipttl[2][i*ND3]+1; 
			ipttl[0][i*ND3+20] = ipttl[0][i*ND3]+2; 
			ipttl[2][i*ND3+20] = ipttl[2][i*ND3]+2; 

			// 21(0,1,2)
			ipttl[0][i*ND3+21] = ipttl[0][i*ND3]+2; 
			ipttl[1][i*ND3+21] = ipttl[1][i*ND3]+1; 
			ipttl[0][i*ND3+22] = ipttl[0][i*ND3]+2; 
			ipttl[1][i*ND3+22] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+22] = ipttl[2][i*ND3]+1; 
			ipttl[0][i*ND3+23] = ipttl[0][i*ND3]+2; 
			ipttl[1][i*ND3+23] = ipttl[1][i*ND3]+1; 
			ipttl[2][i*ND3+23] = ipttl[2][i*ND3]+2; 

			// 22(0,1,2)
			ipttl[0][i*ND3+24] = ipttl[0][i*ND3]+2; 
			ipttl[1][i*ND3+24] = ipttl[1][i*ND3]+2; 
			ipttl[0][i*ND3+25] = ipttl[0][i*ND3]+2; 
			ipttl[1][i*ND3+25] = ipttl[1][i*ND3]+2; 
			ipttl[2][i*ND3+25] = ipttl[2][i*ND3]+1; 
			ipttl[0][i*ND3+26] = ipttl[0][i*ND3]+2; 
			ipttl[1][i*ND3+26] = ipttl[1][i*ND3]+2; 
			ipttl[2][i*ND3+26] = ipttl[2][i*ND3]+2; 
		}
	}

	nttl *= ND3;

	/**
	* read cell type of each cell
	*/
	for (i=0;i<NK/ND;i++) {  
		_lread(hFp,mapCell[i*ND],NI*NJ/ND/ND);
	}
	if (ND == 2) {
		for (k=0;k<NK/ND;k++) {   
			for (j=NJ/ND-1;j>=0;j--) {  
				for (i=NI/ND-1;i>=0;i--) {  
					nCell = *(mapCell[ND*k]+j*NI/ND+i);
					*(mapCell[ND*k]+ND*j*NI+ND*i)= nCell;
					*(mapCell[ND*k]+ND*j*NI+ND*i+1) = nCell;
					*(mapCell[ND*k]+(ND*j+1)*NI+ND*i) = nCell;
					*(mapCell[ND*k]+(ND*j+1)*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+1]+ND*j*NI+ND*i) = nCell;
					*(mapCell[ND*k+1]+ND*j*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+1]+(ND*j+1)*NI+ND*i) = nCell;
					*(mapCell[ND*k+1]+(ND*j+1)*NI+ND*i+1) = nCell;
				}		
			}
		}
		*(mapCell[32*ND]+25*ND*NJ+32*ND) = 3;
		*(mapCell[32*ND]+(25*ND+1)*NJ+32*ND) = 3;
		*(mapCell[32*ND+1]+25*ND*NJ+32*ND) = 3;
		*(mapCell[32*ND+1]+(25*ND+1)*NJ+32*ND) = 3;
	}
	if (ND == 3) {
		for (k=0;k<NK/ND;k++) {   
			for (j=NJ/ND-1;j>=0;j--) {  
				for (i=NI/ND-1;i>=0;i--) {  
					nCell = *(mapCell[ND*k]+j*NI/ND+i);
					*(mapCell[ND*k]+ND*j*NI+ND*i)= nCell;
					*(mapCell[ND*k]+ND*j*NI+ND*i+1) = nCell;
					*(mapCell[ND*k]+ND*j*NI+ND*i+2) = nCell;

					*(mapCell[ND*k]+(ND*j+1)*NI+ND*i) = nCell;
					*(mapCell[ND*k]+(ND*j+1)*NI+ND*i+1) = nCell;
					*(mapCell[ND*k]+(ND*j+1)*NI+ND*i+2) = nCell;

					*(mapCell[ND*k]+(ND*j+2)*NI+ND*i) = nCell;
					*(mapCell[ND*k]+(ND*j+2)*NI+ND*i+1) = nCell;
					*(mapCell[ND*k]+(ND*j+2)*NI+ND*i+2) = nCell;

					*(mapCell[ND*k+1]+ND*j*NI+ND*i)= nCell;
					*(mapCell[ND*k+1]+ND*j*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+1]+ND*j*NI+ND*i+2) = nCell;

					*(mapCell[ND*k+1]+(ND*j+1)*NI+ND*i) = nCell;
					*(mapCell[ND*k+1]+(ND*j+1)*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+1]+(ND*j+1)*NI+ND*i+2) = nCell;

					*(mapCell[ND*k+1]+(ND*j+2)*NI+ND*i) = nCell;
					*(mapCell[ND*k+1]+(ND*j+2)*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+1]+(ND*j+2)*NI+ND*i+2) = nCell;

					*(mapCell[ND*k+2]+ND*j*NI+ND*i)= nCell;
					*(mapCell[ND*k+2]+ND*j*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+2]+ND*j*NI+ND*i+2) = nCell;

					*(mapCell[ND*k+2]+(ND*j+1)*NI+ND*i) = nCell;
					*(mapCell[ND*k+2]+(ND*j+1)*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+2]+(ND*j+1)*NI+ND*i+2) = nCell;

					*(mapCell[ND*k+2]+(ND*j+2)*NI+ND*i) = nCell;
					*(mapCell[ND*k+2]+(ND*j+2)*NI+ND*i+1) = nCell;
					*(mapCell[ND*k+2]+(ND*j+2)*NI+ND*i+2) = nCell;
				}		
			}
		}
		*(mapCell[32*ND]+25*ND*NJ+32*ND) = 3;
		*(mapCell[32*ND]+(25*ND+1)*NJ+32*ND) = 3;
		*(mapCell[32*ND]+(25*ND+2)*NJ+32*ND) = 3;
		*(mapCell[32*ND+1]+25*ND*NJ+32*ND) = 3;
		*(mapCell[32*ND+1]+(25*ND+1)*NJ+32*ND) = 3;
		*(mapCell[32*ND+1]+(25*ND+2)*NJ+32*ND) = 3;
		*(mapCell[32*ND+2]+25*ND*NJ+32*ND) = 3;
		*(mapCell[32*ND+2]+(25*ND+1)*NJ+32*ND) = 3;
		*(mapCell[32*ND+2]+(25*ND+2)*NJ+32*ND) = 3;
	}
	_lclose(hFp);
}



void freemapAPD(void) {
	for (short int i=0;i<NK;i++) {
		free(mapACT[i]);
		free(mapCell[i]);
		free(locXCT[i]);
	}
	free(iparm);
	free(kmin);
	free(kmax);
	free(iStep);
}
void freebrs(void) {
	short int i = 0;
	for (i=0;i<3;i++) {
		free(r[i]);
		free(rn[i]);
	}
	for (i=0;i<NCELL;i++) {
		free(ydata[i]);
	}
}

void freemapAPDcs(void) {
	for (short int i=0;i<NK;i++) {
		free(mapAPD[i]);
		free(mapSpeed[i]); //added by Zhu
	}
}

void freemagcs(void) {
	for (short int i=0;i<4;i++) {
		free(mag[i]);
	}
}

void freePOTcs(void) {
	for(short i=0;i<NL;i++) {
		free(POT_reduce[i]);//by sf 090622
		free(POT[i]);
		free(aw[i]);
	}
	free(POTi);
	free(bw);
}
void freeXCTm(void)
{
	for(short int i=0;i<NCYCL;i++) {
		free(mapXCTm[i]);
	}
}

void freeFibdir(void)
{
	if (ANISO==1) {
		for (short int i=0;i<3;i++) {
			free(fibdir[i]);
		}
	}
}
// read position parameter of heart & call transfer matrix ----
void rdpos(void) {
	void transf(void);

	HFILE hFp;
	short int index;      

	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'p');
	//filepath.SetAt(index+2,'o');
	//filepath.SetAt(index+3,'s');

	//hFp = _lopen(filepath,OF_READ);
	hFp=_lopen(dataPath+"tour.pos ",OF_READ); 
	if (hFp==HFILE_ERROR) {
		fprintf(stdout,"Can not open pos file ! !\n");
		fflush(stdout);
		flag_flop=1;
		return;
	}

	_lread(hFp,&HRTscale,4);
	_lread(hFp,&HRTx0,4);
	_lread(hFp,&HRTy0,4);
	_lread(hFp,&HRTz0,4);
	_lread(hFp,&phai,4);
	_lread(hFp,&pusai,4);
	_lread(hFp,&theta,4);
	_lclose(hFp);

	transf();
}
// Read heart 
/**
* normal cell's para
*/
// APD parameters
// Cell       sn   atr  anv  hb   bb   pkj  vtr  ab1  ab2  ab3  ab4  ab5  ab6  ab7
// Parm       1    2    3    4    5    6    7    8    9    10   11   12   13   14     
//  1  T0     30   10   0    0    0    0    0    0    0    0    0    0    0    0        
//  2  T1     0    0    5    5    5    5    5    5    5    5    5    5    5    5        
//  3  T2     0    0    100  100  100  105  75   75   75   75   75   75   75   75       
//  4  T3     175  120  175  175  175  195  175  175  175  175  175  175  175  175      
//  5  APR    170  100  210  210  210  250  200  200  200  200  200  200  200  200      
//  6  FRT    205  140  320  320  320  345  295  295  295  295  295  295  295  295      
//  7  V0     -90  -90  -90  -90  -90  -90  -90  -90  -90  -90  -90  -90  -90  -90      
//  8  V1     30   -20  40   40   40   40   40   40   40   40   40   40   40   40       
//  9  V2     30   -20  30   30   30   30   30   30   30   30   30   30   30   30       
// 10  GRD    250  0    0    0    0    5    5    5    5    5    5    5    5    5        
// 11  DCS    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 12  DVT    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 13  ECF    100  100  100  100  100  100  100  0    0    0    0    0    0    0        
// 14         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 15         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 16         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 17         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 18  BCL    800  0    0    0    0    0    0    0    0    0    0    0    0    0        
// 19  BN     1    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 20  inc    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 21         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 22         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 23         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 24  ICL    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 25  PRT    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 26  DLY    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 27  ACC    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 28  PBP    0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 29         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 30         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 31         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
// 32  CS     50   100  10   250  250  250  50   -12  0    0    0    0    0    0        
// 33  DC     0    0    0    0    0    0    0    1    0    0    0    0    0    0        
// 34         0    0    0    0    0    1    0    0    0    0    0    0    0    0        
// 35         0    0    0    0    0    0    0    0    0    0    0    0    0    0        
//
void rdAPDm(void)
{
	short int npoint[NCELL],ixsmp[NCELL][100],num;
	float ysmp[NCELL][100];
	short int iT0,iT01,iT012,iT0123;
	short int incr,iBN0,iBCL,iBN,ntstp,iS2ST,iS2CL,iS2BN;
	float dx0,dx1,dx2,dx01,dx02,dx10,dx12,dx20,dx21,a,b;
	HFILE hFp;
	short int icell,i,j,k,icurv,index;
	// hFp=_lopen("f:/apd/apdapd.5",READ);
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'a');
	//filepath.SetAt(index+2,'p');
	//filepath.SetAt(index+3,'d');

	//hFp=_lopen(filepath,OF_READ);
	hFp=_lopen(dataPath+"tour.apd ",OF_READ);

	if(hFp==HFILE_ERROR) {
		fprintf(stdout,"Can not open apd file ! !\n");
		fflush(stdout);
		flag_flop=1;
		return;
	}

	for(icell=0;icell<NCELL;icell++) {
		_lread(hFp,iparm+icell*NPARM,NPARM*2);

		*(iparm+icell*35+0) = *(iparm+icell*35+0)*ND; 
		*(iparm+icell*35+1) = *(iparm+icell*35+1)*ND; 
		*(iparm+icell*35+2) = *(iparm+icell*35+2)*ND; 
		*(iparm+icell*35+3) = *(iparm+icell*35+3)*ND; 
		*(iparm+icell*35+4) = *(iparm+icell*35+4)*ND; 
		*(iparm+icell*35+5) = *(iparm+icell*35+5)*ND; 
		//*(iparm+icell*35+10) = *(iparm+icell*35+10)*ND; 
		_lread(hFp,&npoint[icell],2);
		for(j=0;j<npoint[icell];j++) {
			_lread(hFp,&ixsmp[icell][j],2);
			ixsmp[icell][j] = ixsmp[icell][j]*ND;
			_lread(hFp,&ysmp[icell][j],4);
		}
	}
	_lclose(hFp);

	mBCL=0;
	miBN=0;
	mS2ST=0;
	mS2CL=0;
	mS2BN=0;
	maxXctStep=0; 
	for(icell=0;icell<NCELL;icell++) {
		incr=*(iparm+icell*NPARM+19);
		if(incr<0) {
			iBN0=1-100/incr;
			if(*(iparm+icell*NPARM+18)>iBN0) *(iparm+icell*NPARM+18)=iBN0;
		}
		//iBCL=*(iparm+icell*NPARM+17); //basic cycle length
		*(iparm+icell*NPARM+17) = *(iparm+icell*NPARM+17)*ND;
		// S2, additional stimulus
		*(iparm+icell*NPARM+14) = *(iparm+icell*NPARM+14)*ND;
		*(iparm+icell*NPARM+15) = *(iparm+icell*NPARM+15)*ND;
		iS2ST=*(iparm+icell*NPARM+14);
		iS2CL=*(iparm+icell*NPARM+15);
		iS2BN=*(iparm+icell*NPARM+16);
		iBCL=*(iparm+icell*NPARM+17); //basic cycle length
		iBN=*(iparm+icell*NPARM+18); // beat number
		ntstp=(iBN*iBCL+iBN*(iBN-1)*iBCL*incr/200)/3;
		if(iBCL>mBCL) mBCL=iBCL;
		if(iBN>miBN) miBN=iBN;
		if(iS2ST>mS2ST) mS2ST=iS2ST;
		if(iS2CL>mS2CL) mS2CL=iS2CL;
		if(iS2BN>mS2BN) mS2BN=iS2BN;
		if(ntstp>maxXctStep) maxXctStep=ntstp;
		// CL increament: % --> TS
		*(iparm+icell*NPARM+19)=iBCL*incr*ND/300;
		*(iparm+icell*NPARM+17)=iBCL/3;
		// iparm(18) <-- total pacing time
		*(iparm+icell*NPARM+18)=ntstp;
		// FRP <-- FRP-ARP
		*(iparm+icell*NPARM+5)=*(iparm+icell*NPARM+5)-*(iparm+icell*NPARM+4);
		// intrinsic CL: ms --> TS
		*(iparm+icell*NPARM+23)=*(iparm+icell*NPARM+23)/3;
		//-- conduction speed(100*)CS:
		//    CS(m/s) --> CS(2*1.5 mm/3ms) --> CS*2(cell/Step) ----  
		/*<Comment by ALF> why 100*/
		*(iparm+icell*NPARM+31)=*(iparm+icell*NPARM+31)*2;
		// we only have two points to represent his bundle and bundle branches
		//if (icell == 3) *(iparm+(icell-1)*NPARM+31)=*(iparm+(icell-1)*NPARM+31)/11;
		//if (icell == 3) *(iparm+(icell-1)*NPARM+31)=100;
		//if (icell == 5) *(iparm+(icell-1)*NPARM+31)=*(iparm+(icell-1)*NPARM+31)/ND;
		// initialize ydata
		for(short int n=0;n<1000*ND;n++)
			*(ydata[icell]+n)=(float)*(iparm+icell*NPARM+6);
		//for (int ii=0; ii < NPARM; ii++ ) 
		//TRACE("\nCell %2d %2d %d", icell, ii, *(iparm+icell*NPARM+ii));
	}

	// --- data set ---
	for(icurv=0;icurv<NCELL;icurv++) {
		num=npoint[icurv];
		iT0=*(iparm+icurv*NPARM);
		iT01=iT0+*(iparm+icurv*NPARM+1);
		iT012=iT01+*(iparm+icurv*NPARM+2);
		iT0123=iT012+*(iparm+icurv*NPARM+3);
		//---- lenth of APD ------
		la012[icurv]=iT012;
		la0123[icurv]=iT0123;
		// --- t =  phased 0 ---
		for(i=0;i<=(iT0-1);i++) { // < ? July 4, 1996
			//    +++++ iparm(icurv,6), the real value +++++
			a=(float)(-*(iparm+icurv*NPARM+6)+*(iparm+icurv*NPARM+7))
				/(float)*(iparm+icurv*NPARM);
			b=(float)*(iparm+icurv*NPARM+6);
			*(ydata[icurv]+i)=a*i+b;
		}
		*(ydata[icurv]+iT0)=(float)*(iparm+icurv*NPARM+7);

		// --- t =  phase 1 ---
		if(iT01>iT0) {
			for(i=(iT0+1);i<=(iT01-1);i++) {
				a=(float)(*(iparm+icurv*NPARM+8)-*(iparm+icurv*NPARM+7))
					/(float)*(iparm+icurv*NPARM+1);
				b=(float)*(iparm+icurv*NPARM+7)-a*iT0;
				*(ydata[icurv]+i)=a*i+b;
			}
		}
		// --- t = phase 2 ---
		for(i=iT01;i<=iT012;i++)
			*(ydata[icurv]+i)=(float)*(iparm+NPARM*icurv+8);
		//---- t= phase 3 ----

		for(i=(iT012+1);i<=iT0123;i++) {
			if((i<ixsmp[icurv][num-3])&&(i>ixsmp[icurv][num-2])) {
				dx0=(float)(i-ixsmp[icurv][num-1]);
				dx1=(float)(i-ixsmp[icurv][num-2]);
				dx2=(float)(i-ixsmp[icurv][num-3]);
				dx01=(float)(ixsmp[icurv][num-1]-ixsmp[icurv][num-2]);
				dx02=(float)(ixsmp[icurv][num-1]-ixsmp[icurv][num-3]);
				dx10=(float)(ixsmp[icurv][num-2]-ixsmp[icurv][num-1]);
				dx12=(float)(ixsmp[icurv][num-2]-ixsmp[icurv][num-3]);
				dx20=(float)(ixsmp[icurv][num-3]-ixsmp[icurv][num-1]);
				dx21=(float)(ixsmp[icurv][num-3]-ixsmp[icurv][num-2]);
				*(ydata[icurv]+i)=dx1*dx2*ysmp[icurv][num-1]/dx01/dx02
					+dx0*dx2*ysmp[icurv][num-2]/dx10/dx12
					+dx0*dx1*ysmp[icurv][num-3]/dx20/dx21;
			}

			for(k=2;k<num-3;k++) {
				if(i==ixsmp[icurv][k+1]) 
					*(ydata[icurv]+i)=ysmp[icurv][k+1];
				else if(i==ixsmp[icurv][k]) 
					*(ydata[icurv]+i)=ysmp[icurv][k];
				else if((i<ixsmp[icurv][k])&&(i>ixsmp[icurv][k+1])) {
					dx0=(float)(i-ixsmp[icurv][k+1]);
					dx1=(float)(i-ixsmp[icurv][k]);
					dx2=(float)(i-ixsmp[icurv][k-1]);
					dx01=(float)(ixsmp[icurv][k+1]-ixsmp[icurv][k]);
					dx02=(float)(ixsmp[icurv][k+1]-ixsmp[icurv][k-1]);
					dx10=(float)(ixsmp[icurv][k]-ixsmp[icurv][k+1]);
					dx12=(float)(ixsmp[icurv][k]-ixsmp[icurv][k-1]);
					dx20=(float)(ixsmp[icurv][k-1]-ixsmp[icurv][k+1]);
					dx21=(float)(ixsmp[icurv][k-1]-ixsmp[icurv][k]);
					*(ydata[icurv]+i)=dx1*dx2*ysmp[icurv][k+1]/dx01/dx02
						+dx0*dx2*ysmp[icurv][k]/dx10/dx12
						+dx0*dx1*ysmp[icurv][k-1]/dx20/dx21;
				}            
			}
		}
	}
}




/**
* transform matrix for (i,j,k) -> (x, y, z)
*/
// transf: coordinate transformation 
void transf(void) {
	short int i,j,k;  
	float a2[3][3],a;
	float a1[3][6]={
		1.0,  0.5,    0.5, -0.5,   -0.5,    0.0,
		0.0,0.866, 0.2886,0.866, 0.2886,-0.5773,
		0.0,  0.0,-0.8165,  0.0,-0.8165,-0.8165}; 
		float rd=1.745329252e-2;
		float ph=rd*phai;
		float ps=rd*pusai;
		float th=rd*theta;
		float cph=cos(ph);
		float sph=sin(ph);
		float cps=cos(ps);
		float sps=sin(ps);
		float cth=cos(th);
		float sth=sin(th);

		a2[0][0]=cps*cph-cth*sps*sph;
		a2[0][1]=-sps*cph-cth*cps*sph;
		a2[0][2]=sth*sph;
		a2[1][0]=cps*sph+cth*sps*cph;
		a2[1][1]=-sps*sph+cth*cps*cph;
		a2[1][2]=-sth*cph;
		a2[2][0]=sps*sth;
		a2[2][1]=cps*sth;
		a2[2][2]=cth;

		for (i=0;i<3;i++) {
			for (j=0;j<6;j++) {
				a=0;
				for (k=0;k<3;k++)
					a=a+a2[i][k]*a1[k][j];
				tmswf[i][j]=(float)(a*HRTscale/ND);
			}
		}
}
/**
* torso position
*/
// Read the data of nodes and derivatives
void rdnod(void) {
	short int i;
	HFILE hFp;
	//short int index;

	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'n');
	//filepath.SetAt(index+2,'o');
	//filepath.SetAt(index+3,'d');
	//hFp=_lopen(filepath,OF_READ);  
	hFp=_lopen(dataPath+"tour.nod ",OF_READ);
	if (hFp==HFILE_ERROR) {
		fprintf(stdout,"Can not open nod file ! !\n");
		flag_flop=1;
		return;
	}

	_lread(hFp,&nPos,2);

	if (nPos>NL) nPos=NL;

	for (i=0;i<nPos;i++) {
		_lread(hFp,r[0]+i,4);
		_lread(hFp,r[1]+i,4);
		_lread(hFp,r[2]+i,4);
	}

	for (i=0;i<nPos;i++) {
		_lread(hFp,rn[0]+i,4);
		_lread(hFp,rn[1]+i,4);
		_lread(hFp,rn[2]+i,4);
	}
	_lclose(hFp);

}

// Read electrode position file
void rdelc(void) {
	short int i;
	float eps[3][6],weight[3][6];
	short int member[3][6];

	HFILE hFp;      
	//short int index;
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'e');
	//filepath.SetAt(index+2,'l');
	//filepath.SetAt(index+3,'c');

	//hFp=_lopen(filepath,OF_READ); 
	hFp=_lopen(dataPath+"tour.elc ",OF_READ);
	if (hFp==HFILE_ERROR) {
		fprintf(stdout,"Can not open elc file ! !\n");
		flag_flop=1;
		return;
	}

	for (i=0;i<6;i++) {
		_lread(hFp,&eps[0][i],4);
		_lread(hFp,&eps[1][i],4);
		_lread(hFp,&eps[2][i],4);
	}

	for (i=0;i<6;i++) {
		_lread(hFp,&weight[0][i],4);
		_lread(hFp,&weight[1][i],4);
		_lread(hFp,&weight[2][i],4);
	}

	for(i=0;i<6;i++) {
		_lread(hFp,&member[0][i],2);
		_lread(hFp,&member[1][i],2);
		_lread(hFp,&member[2][i],2);
	}
	_lread(hFp,&nv[0],2);
	_lread(hFp,&nv[1],2);
	_lread(hFp,&nv[2],2);
	_lclose(hFp);
}

void locfile(void) {
	short int i,j,k;
	totalCell=0;
	for (k=0;k<NK;k++)    
		for (j=0;j<NJ;j++)   
			for (i=0;i<NI;i++) {  
				if ((*(mapCell[k]+j*NI+i)>0)&&(*(mapCell[k]+j*NI+i)<NCELL+1)) {
					*(locXCT[k]+j*NI+i)=totalCell;
					totalCell++;
				} else *(locXCT[k]+j*NI+i)=-1;    
			}
			//TRACE("\nTotal Cells: %d", totalCell);
} 
// Geometric information of heart model 
void geoinfc(void) {
	int i0, ii, endoAn, iendo;
	short int i,j,k;
	short int l, m, flag;
	short int endoAx[20000*ND3];
	short int endoAy[20000*ND3];
	short int endoAz[20000*ND3];

	//short int iseqx[12]={-1,-1, 0, 0, 1, 0, 1, 1, 0, 0,-1, 0 };
	//short int iseqy[12]={ 0, 1, 1, 0, 0, 1, 0,-1,-1, 0, 0,-1 };
	//short int iseqz[12]={ 0, 0, 0,-1,-1,-1, 0, 0, 0, 1, 1, 1 };

	/**
	* coor-delta matrix
	*/
	short int iseqx[6]={-1, 0, 0, 1, 0, 0}; 
	short int iseqy[6]={ 0, 1, 0, 0,-1, 0};
	short int iseqz[6]={ 0, 0,-1, 0, 0, 1};

	// Margins of each (i,j)

	/**
	* max_min value of k of model at (i,j)
	*/
	// get kmin and kmax for each [NI][NJ] frame
	for (i=0;i<NI;i++) {
		for (j=0;j<NJ;j++) {
			k = 0;
			while (k < NK) {
				if (*(mapCell[k]+j*NI+i)>0) { /*<Comment by ALF> have cell, some duplicate point by using this method*/
					*(kmin+j*NI+i)=k;
					for (k=NK-1;k>-1;k--) {
						if (*(mapCell[k]+j*NI+i)>0) {
							*(kmax+j*NI+i)=k;
							k = NK*2;
							break;
						}
					}
				} 
				k++;
			}
			if (k < NK*2) {
				*(kmin+j*NI+i)=NK+1;
				*(kmax+j*NI+i)=0;
			}
		}
	}

	//-------------------- modified by ALF at 2008-8-19 begin -------------------->
	//add: get epicardial triangle's vertex position, also some duplicate point in epicXYZ
	//Nepic = NI*NJ*2; //by sf
	epicX.reserve(Nepic);
	epicY.reserve(Nepic);
	epicZ.reserve(Nepic);

	for (i=0; i<NI; ++i) {
		for (j=0; j<NJ; ++j) {
			epicX.push_back(i);
			epicY.push_back(j);
			epicZ.push_back(*(kmin+j*NI+i)-1);
		}
	}
	for (i=0; i<NI; ++i) {
		for (j=0; j<NJ; ++j) {
			epicX.push_back(i);
			epicY.push_back(j);
			epicZ.push_back(*(kmax+j*NI+i)+1);
		}
	}
	for (i=0; i<Nepic; ++i) {
		epicX_old[i]=epicX[i];
		epicY_old[i]=epicY[i];
		epicZ_old[i]=epicZ[i];
	}
	//-------------------- modified by ALF at 2008-8-19 end --------------------< 

	// get kTop: minimum of kmin and 
	//     kBtm: maximum of kmax
	kTop=NK+1; 
	kBtm=0;
	for (i=0;i<NI;i++) {
		for (j=0;j<NJ;j++) {
			if (*(kmin+j*NI+i)<kTop) kTop=*(kmin+j*NI+i);
			if (*(kmax+j*NI+i)>kBtm) kBtm=*(kmax+j*NI+i);
		}
	}
	// get kVtr: ventricular position, so heart can be divided into two parts
	for (k=kTop;k<=kBtm;k++) {
		for (i=0;i<NI;i++) {
			for (j=0;j<NJ;j++) {
				if ((*(mapCell[k]+j*NI+i)>4)&&(*(mapCell[k]+j*NI+i)<15)) {
					kVtr=k; // ventricular position
					i = NI;
					j = NJ;
					k = kBtm;
				}
			}
		}
	}

	nHB=0;
	nBB=0;
	// get Bundle branches & his branches' position
	for (k=kTop;k<=kBtm;k++) {
		for (i=0;i<NI;i++) {
			for (j=0;j<NJ;j++) {
				//if (*(mapCell[k]+j*NI+i)==4) {
				// change to BB
				if (*(mapCell[k]+j*NI+i)==5) {
					iBB[0][nBB]=i;
					iBB[1][nBB]=j;
					iBB[2][nBB]=k;
					nBB++;
				} else if (*(mapCell[k]+j*NI+i)==4) {
					iHB[0][nHB]=i;
					iHB[1][nHB]=j;
					iHB[2][nHB]=k;
					nHB++;
				}

			}
		}
	}
	// get endocardial positions
	for (m=0;m<2;m++) {
		for (ii=0;ii<20000*ND3;ii++) {
			endoAx[ii]=0;
			endoAy[ii]=0;
			endoAz[ii]=0;
		}
		if (m==0) {
			endoAx[0]=24*ND;
			endoAy[0]=30*ND;
			endoAz[0]=40*ND;
			for (ii=0;ii<NENDO*ND3;ii++) {
				endoBx[ii]=0;
				endoBy[ii]=0;
				endoBz[ii]=0;
			}
		} else if (m==1) {
			endoAx[0]=26*ND;
			endoAy[0]=13*ND;
			endoAz[0]=36*ND;
			for (ii=0;ii<NENDO*ND3;ii++) {
				endoCx[ii]=0;
				endoCy[ii]=0;
				endoCz[ii]=0;
			}
		}
		// TRACE("\nFirst %d",*(mapCell[endoAz[0]]+endoAy[0]*NI+endoAx[0]));
		*(mapCell[endoAz[0]]+endoAy[0]*NI+endoAx[0])=30;
		iendo=0;
		endoAn=1;
		i0=0;	
		while (i0<endoAn) {
			flag=0;
			for (l=0;l<6;l++) {
				i=endoAx[i0]+iseqx[l];
				if((i<0)||(i>NI)) continue;
				j=endoAy[i0]+iseqy[l];
				if((j<0)||(j>NJ)) continue;
				k=endoAz[i0]+iseqz[l];
				if((k<kTop)||(k>kBtm)) continue;
				if (*(mapCell[k]+j*NI+i)==0) { /*<Comment by ALF> find the normal direction */
					*(mapCell[k]+j*NI+i)=30;  /*<Comment by ALF> 30 is only a value to make sure no confuse with valid type */
					endoAx[endoAn]=i;
					endoAy[endoAn]=j;
					endoAz[endoAn]=k;
					endoAn++;
				}
				if ((flag==0) && *(mapCell[k]+j*NI+i)>0 && *(mapCell[k]+j*NI+i)<16) {
					if (m==0) {
						endoBx[iendo]=endoAx[i0];
						endoBy[iendo]=endoAy[i0];
						endoBz[iendo]=endoAz[i0];
					} else if (m==1) {
						endoCx[iendo]=endoAx[i0];
						endoCy[iendo]=endoAy[i0];
						endoCz[iendo]=endoAz[i0];					
					}
					iendo++;
					flag=1;
				}
			}
			i0++;
		}
		if  (m==0) {
			NendoB=iendo;
			//TRACE("\nEndo B %d",NendoB);
		} else if (m==1) { 
			NendoC=iendo;
			//TRACE("\nEndo C %d",NendoC);
		}
	}

	for (k=kTop;k<=kBtm;k++) {
		for (i=0;i<NI;i++) {
			for (j=0;j<NJ;j++) {
				if (*(mapCell[k]+j*NI+i)==30) {
					*(mapCell[k]+j*NI+i)=0;
				}
			}
		}
	}
}


// ---- set parameter of the anisotropy ------
//
//   velocicy(l)=0.5 m/s ==> dist*3/9msec
//   velocity(t)/velocity(l)=1/3
//   velocity(t)/velocity(l)=0.42?
//   resistance(t)/(l)=9
//       according to Clerc 
//       see Robert D.E., Circ. Res. 44:701-712,1979
//   Input: dist
//   Output: vl2[10],vt2[10],rrat1
//
void setaniso(void) {
	short int i, ltrat;
	float vl,vt,vrat,rrat;
	float fct; 

	ltrat=2;
	//ltrat=1;
	fct=1.1;
	vrat=1.0/ltrat;
	rrat=1.0/9;
	//vrat=1.0;
	//rrat=1.0;
	rrat1=rrat-1.;
	for (i=0; i<10;i++) {
		vl=fct*(i+1)*HRTscale/ND;
		vt=vl*vrat;
		vl2[i]=vl*vl;
		vt2[i]=vt*vt;
		//TRACE("\ni vl2 vt2 %2d %f %f,", i, vl2[i],vt2[i]);
	}
}

//
// --- calculate out-products of 'cell-neighber vectors' ----
//
//
void neibdir (void) {
	void ijktoxyz(short int [3], float [3]);
	short int i, j;
	short int istrt[3],iterm[3],iterm1[3];
	float strt[3],term[3],dir[3];
	float term1[3],dir1[3],r;

	short int iseqx[12]={-1,-1, 0, 0, 1, 0, 1, 1, 0, 0,-1, 0 };
	short int iseqy[12]={ 0, 1, 1, 0, 0, 1, 0,-1,-1, 0, 0,-1 };
	short int iseqz[12]={ 0, 0, 0,-1,-1,-1, 0, 0, 0, 1, 1, 1 };
	istrt[0]=0;
	istrt[1]=0;
	istrt[2]=0;
	ijktoxyz(istrt,strt);

	for (i=0;i<12;i++) 
		for (j=0;j<12;j++) {
			if (i==j) { 
				prx[i][j]=0.;
				pry[i][j]=0.;
				prz[i][j]=0.;
				continue;
			}
			if (i>5 && j>5) {
				prx[i][j]=prx[i-6][j-6];
				pry[i][j]=pry[i-6][j-6];
				prz[i][j]=prz[i-6][j-6];
				continue;
			}
			if (j>5) {
				prx[i][j]=-prx[i][j-6];
				pry[i][j]=-pry[i][j-6];
				prz[i][j]=-prz[i][j-6];
				continue;
			}
			if (i>5) {
				prx[i][j]=-prx[i-6][j];
				pry[i][j]=-pry[i-6][j];
				prz[i][j]=-prz[i-6][j];
				continue;
			}
			iterm[0]=iseqx[i];
			iterm[1]=iseqy[i];
			iterm[2]=iseqz[i];
			ijktoxyz(iterm,term);
			//linedir(strt,term,dir);
			dir[0]=term[0]-strt[0];
			dir[1]=term[1]-strt[1];
			dir[2]=term[2]-strt[2];
			r=sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
			dir[0]=dir[0]/r;
			dir[1]=dir[1]/r;
			dir[2]=dir[2]/r;        
			iterm1[0]=iseqx[j];
			iterm1[1]=iseqy[j];
			iterm1[2]=iseqz[j];
			ijktoxyz(iterm1,term1);
			//linedir(strt,term1,dir1);
			dir1[0]=term1[0]-strt[0];
			dir1[1]=term1[1]-strt[1];
			dir1[2]=term1[2]-strt[2];
			r=sqrt(dir1[0]*dir1[0]+dir1[1]*dir1[1]+dir1[2]*dir1[2]);
			dir1[0]=dir1[0]/r;
			dir1[1]=dir1[1]/r;
			dir1[2]=dir1[2]/r;        
			//TRACE("\nlidir1 %f %f %f ",dir1[0],dir1[1],dir1[2]);
			// outprod(dir,dir1,out);
			prx[i][j]=dir[1]*dir1[2]-dir[2]*dir1[1];     
			pry[i][j]=dir[2]*dir1[0]-dir[0]*dir1[2];
			prz[i][j]=dir[0]*dir1[1]-dir[1]*dir1[0];
			r=sqrt(prx[i][j]*prx[i][j]+pry[i][j]*pry[i][j]+prz[i][j]*prz[i][j]);
			prx[i][j]=prx[i][j]/r;
			pry[i][j]=pry[i][j]/r;
			prz[i][j]=prz[i][j]/r;
		}
		/*
		for (i=0;i<6;i++) 
		for (j=0;j<6;j++) {
		TRACE("\nneibdir %d %d %f %f %f ",i,j,prx[i][j],pry[i][j],prz[i][j]);
		} 
		*/   
}


//
// ----  fibplane direction angle ------
//      all plane directions are in j=22 (assumed to be parallel
//      to the septal plane
//      for all directions, lines atart from (1,22,90) to
//      (Note: Selectable)plane(1): (50,22,90) 
//                      (assumed perpendicular to heart axis)
//
void fibplane (void) {
	float getAngle(float [], float []);
	void ijktoxyz(short int [], float []);
	//void linedir(float [], float [], float []);

	short int i, j, k, n;
	short int iorg[3]={1,19,90};
	short int iterm0[3]={1,19,1};
	short int iterm[3];

	float org[3];
	float term0[3],term[3];
	float dir0[3],dir[3];
	float r;
	float ang=1.;
	float arch=1.;
	float pai=3.1415926;
	float delt;

	// ---- angle per layer, max rotation angle=pi/2 
	//TRACE("\nmaxlayer= %d ",maxlay);
	arch=pai/180.;
	if (maxlay<=0) return;
	//delt=(2./3.)*pai/maxlay;
	delt=(1./4.)*pai/maxlay;
	//TRACE("\ndelt/arch= %f ",delt/arch);
	// ----- all in septal plane ----->
	ijktoxyz(iorg,org);
	ijktoxyz(iterm0,term0);
	//linedir(org,term0,dir0);
	dir0[0]=term0[0]-org[0];
	dir0[1]=term0[1]-org[1];
	dir0[2]=term0[2]-org[2];
	r=sqrt(dir0[0]*dir0[0]+dir0[1]*dir0[1]+dir0[2]*dir0[2]);
	planedir[0][0]=dir0[0]/r;
	planedir[1][0]=dir0[1]/r;
	planedir[2][0]=dir0[2]/r;
	// --- search next planedir ---->
	i=iterm0[0];
	j=iterm0[1];
	k=iterm0[2];

	// TRACE("\nplanedir  0 %f %f %f",planedir[0][0],planedir[1][0],planedir[2][0]);
	for (n=1; n<=maxlay; n++) {
		do {
			if (i<NI) {
				i=i+1;
			} else { 
				k=k+1;
			}
			iterm[0]=i;
			iterm[1]=j;
			iterm[2]=k;
			ijktoxyz(iterm,term);
			//linedir(org,term,dir);
			dir[0]=term[0]-org[0];
			dir[1]=term[1]-org[1];
			dir[2]=term[2]-org[2];
			r=sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
			dir[0]=dir[0]/r;
			dir[1]=dir[1]/r;
			dir[2]=dir[2]/r;        
			ang=getAngle(dir0,dir);
			//TRACE("\n %f %f %f %f %f %f %f %d %d %d %f %f",dir0[0],dir0[1],dir0[2],
			//	dir[0],dir[1],dir[2],ang,i,j,k,ang*arch,n*delt);
		} while (ang*arch < n*delt);
		planedir[0][n]=dir[0];
		planedir[1][n]=dir[1];
		planedir[2][n]=dir[2];
		//TRACE("\nplanedir %2d %f %f %f",n,planedir[0][n],planedir[1][n],planedir[2][n]);
	}
	// for test ---->
	/*
	for (n=0; n< maxlay; n++) {
	for (m=0; m<3; m++) {
	dir0[m]=planedir[m][n];
	dir[m]=planedir[m][n+1];
	}
	ang=getAngle(dir0,dir);
	TRACE("\nn,ang %d,%f",n,ang);
	}
	*/
	// <---- test end             
}

//
//******** fiber direction for each (i,j,k) *************
//
void fibdirct(void) {
	float getAngle(float [], float []);
	short int normdir(short int, short int, short int, float []);
	short int i, j, k, nneib, iLayer;
	char iCell;

	int locfib;
	// float test[3],test1[3];
	float tmpx,tmpy,tmpz;
	// float ang;
	float pdirx,pdiry,pdirz,r;
	//float dirx,diry,dirz;
	float nordir[3];

	//dirx=0.;
	//diry=0.;
	//dirz=0.;
	for(k=0;k<NK;k++) { 
		for(j=0;j<NJ;j++) {
			for(i=0;i<NI;i++) {
				locfib=*(locXCT[k]+j*NI+i);
				iCell=*(mapCell[k]+j*NI+i);
				if (iCell!=7) iLayer=0;
				else iLayer=*(mapACT[k]+j*NI+i)+1;
				if (locfib==-1) continue;
				//if (*(MapLyr+locfib)<=0) continue;
				//if (*(MapLyr+locfib) >= 30) continue;
				if (iLayer<=0 || iLayer>=30) continue;
				nneib=normdir(i,j,k,nordir);
				r=sqrt(nordir[0]*nordir[0]+nordir[1]*nordir[1]+nordir[2]*nordir[2]);
				//TRACE("\nnordir %2d %2d %2d %f %f %f %d", 
				//	i+1,j+1,k+1,nordir[0],nordir[1],nordir[2],nneib); 
				if (r<0.0000001) continue;                
				pdirx=planedir[0][iLayer-1];
				pdiry=planedir[1][iLayer-1];
				pdirz=planedir[2][iLayer-1];
				//TRACE("\npdir %f %f %f %d",pdirx,pdiry,pdirz,iLayer); 
				// --- fiberdir = planedir X normldir
				tmpx=pdiry*nordir[2]-pdirz*nordir[1];
				tmpy=pdirz*nordir[0]-pdirx*nordir[2];
				tmpz=pdirx*nordir[1]-pdiry*nordir[0];
				r=sqrt(tmpx*tmpx+tmpy*tmpy+tmpz*tmpz);
				if (r<0.0000001) continue;                
				*(fibdir[0]+locfib)=tmpx/r;
				*(fibdir[1]+locfib)=tmpy/r;
				*(fibdir[2]+locfib)=tmpz/r;
				//TRACE("\nfibdir %2d %2d %2d %f %f %f %d",i+1,j+1,k+1, *(fibdir[0]+locfib),
				//	*(fibdir[1]+locfib),*(fibdir[2]+locfib),locfib); 
			}
		}
	}

	// ---- for test------>
	/*
	TRACE("\nj=22"); 
	i=21;
	for(k=58;k<62;k++) { 
	for(j=31;j<33;j++) {
	for(n=0;n<3;n++) {
	locfib=*(locXCT[k]+j*NI+i);
	test[n]=*(fibdir[n]+locfib);
	test1[n]=planedir[n][1];
	}
	ang=getAngle(test,test1);
	TRACE("\n %d %d %d %f %f %f %f %d", 
	i,j,k,ang,test[0],test[1],test[2],locfib); 
	}
	}
	*/
}

//
//  calculate normal direction of fibplane at cell i
//
short int normdir(short int icl,short int jcl,short int kcl, 
	float nordir[3]){
		char iCell,jCell;
		short int i,iLayer,jLayer;
		short int iface[12];
		int locnor, jloc;
		short int jx,jy,jz,l;
		float r,dirx,diry,dirz; 
		short int nneib;
		short int iseqx[12]={-1,-1, 0, 0, 1, 0, 1, 1, 0, 0,-1, 0};
		short int iseqy[12]={ 0, 1, 1, 0, 0, 1, 0,-1,-1, 0, 0,-1};
		short int iseqz[12]={ 0, 0, 0,-1,-1,-1, 0, 0, 0, 1, 1, 1};

		nneib=0;
		r=0.;
		nordir[0]=0.;
		nordir[1]=0.;
		nordir[2]=0.;
		dirx=0.;
		diry=0.;
		dirz=0.;
		for (i=0;i<3;i++) {
			nordir[i]=0.;
		}
		for (i=0;i<12;i++) {
			iface[i]=0;
		}
		locnor=*(locXCT[kcl]+jcl*NI+icl);
		iCell=*(mapCell[kcl]+jcl*NI+icl);
		if (iCell!=7) iLayer=0;
		else iLayer=*(mapACT[kcl]+jcl*NI+icl)+1;
		for (l=0;l<12;l++) {
			jx=icl+iseqx[l];
			if(jx<0 || jx>=NI) continue;
			jy=jcl+iseqy[l];
			if(jy<0 || jy>=NJ) continue;
			jz=kcl+iseqz[l];
			if(jz<0 || jz>=NK) continue;
			jloc=*(locXCT[jz]+jy*NI+jx);
			if(jloc==-1) continue;
			jCell=*(mapCell[jz]+jy*NI+jx);
			if (jCell!=7) jLayer=0;
			else jLayer=*(mapACT[jz]+jy*NI+jx)+1;
			//TRACE("\n%2d %2d %2d %d %d %d %d", jx+1,jy+1,jz+1,jCell,jLayer,iLayer,nneib);
			if(jLayer<1) continue;
			if(iLayer!=jLayer) continue;
			iface[nneib]=l;
			nneib=nneib+1;
		}
		// --- neglect fiber edge --->
		if(nneib<=1) return nneib;
		for(l=0;l<nneib-1;l++) {
			dirx=dirx+prx[iface[l]][iface[l+1]];
			diry=diry+pry[iface[l]][iface[l+1]];
			dirz=dirz+prz[iface[l]][iface[l+1]];
		}
		// --- two neighbering points only --->
		// --- in opposite --->
		dirx=dirx+prx[iface[nneib-1]][iface[0]];
		diry=diry+pry[iface[nneib-1]][iface[0]];
		dirz=dirz+prz[iface[nneib-1]][iface[0]];

		dirx=dirx/(1.*nneib);
		diry=diry/(1.*nneib);
		dirz=dirz/(1.*nneib);
		r=sqrt(dirx*dirx+diry*diry+dirz*dirz);
		if (r<0.00001) {
			;//TRACE("\nicl,jcl,kcl,nneib,iface %d %d %d %d %d %d", 
			//	icl,jcl,kcl,nneib,iface[0],iface[1]);
		} else {
			dirx=dirx/r;
			diry=diry/r;
			dirz=dirz/r;
		}
		nordir[0]=dirx;
		nordir[1]=diry;
		nordir[2]=dirz;
		return nneib;
}
//
//  ---- angle of two vectors ---
//
float getAngle (float vct1[3], float vct2[3]) {
	short int n;
	float pi=3.1415926;
	float ang1=0.;
	float sumv=0.;
	float sumv1=0.;
	float sumv2=0.;
	for (n=0;n<3;n++) {
		sumv1=sumv1+vct1[n]*vct1[n];
		sumv2=sumv2+vct2[n]*vct2[n];
		sumv=sumv+vct1[n]*vct2[n];
	}
	ang1=acos(sumv/sqrt(sumv1*sumv2))* 180. / pi;
	return ang1;
}

//
// ----transform from I,J,K to Z,Y,Z -------
//
void ijktoxyz(short int ijk[3], float xyz[3]) {
	xyz[0]=HRTx0+ijk[0]*tmswf[0][0]+ijk[1]*tmswf[0][1]+ijk[2]*tmswf[0][2];
	xyz[1]=HRTy0+ijk[0]*tmswf[1][0]+ijk[1]*tmswf[1][1]+ijk[2]*tmswf[1][2];
	xyz[2]=HRTz0+ijk[0]*tmswf[2][0]+ijk[1]*tmswf[2][1]+ijk[2]*tmswf[2][2];
}

//
// ----transform coordinate system to I,J,K to establish
//        local coordinate system (shift and rotated) -------
//
//     step1: shift old system to i,j,k 
//            the old coordinate system
//     step2: rotate x,y,z system to fiber coordinate
//            system so that Z axis has direction of
//            fibdir(i,j,k) and X axis has direction
//            fiber direction and Y axis=Z(x)X
//     step3: solve equation
//            x=l1*X+l2*Y+l3*Z
//            y=m1*X+m2*Y+m3*Z
//            z=n1*X+n2*Y+n3*Z
//            where l,m,n is the dirction number of axises
//            l=cos(alpha), m=cos(beta) and n=cos(theta)
//
float local(short int i, short int j, short int k) {
	// ++++ d as a mark of whether the trasform is successful +++
	float getAngle(float [], float []);

	char iCell;
	short int n,iLayer;
	int locloc;
	float r,d;

	// -- step 1 --->
	//	 x=ijk(1)*tmswf(1,1)+ijk(2)*tmswf(1,2)+ijk(3)*tmswf(1,3)
	//   y=ijk(1)*tmswf(2,1)+ijk(2)*tmswf(2,2)+ijk(3)*tmswf(2,3)
	//   z=ijk(1)*tmswf(3,1)+ijk(2)*tmswf(3,2)+ijk(3)*tmswf(3,3)
	// ---step 2 : Y axis ---->
	//
	locloc=*(locXCT[k]+NI*j+i);
	iCell=*(mapCell[k]+NI*j+i);
	if (iCell!=7) iLayer=0;
	else iLayer=*(mapACT[k]+j*NI+i)+1;
	for (n=0; n<3; n++) {
		zaxis[n]=*(fibdir[n]+locloc);
		xaxis[n]=planedir[n][iLayer-1];
	}
	if (zaxis[0]<0.0000001 && zaxis[0]>-0.0000001 &&
		zaxis[1]<0.0000001 && zaxis[1]>-0.0000001 &&
		zaxis[2]<0.0000001 && zaxis[2]>-0.0000001 ) {
			d=0.;
			return d;
	}
	// call outprod(zaxis,xaxis,yaxis)
	yaxis[0]=zaxis[1]*xaxis[2]-zaxis[2]*xaxis[1];     
	yaxis[1]=zaxis[2]*xaxis[0]-zaxis[0]*xaxis[2];
	yaxis[2]=zaxis[0]*xaxis[1]-zaxis[1]*xaxis[0];
	r=sqrt(yaxis[0]*yaxis[0]+
		yaxis[1]*yaxis[1]+yaxis[2]*yaxis[2]);
	if (r < 0.0000001) {
		d=0.;
		return d;
	}
	yaxis[0]=yaxis[0]/r;
	yaxis[1]=yaxis[1]/r;
	yaxis[2]=yaxis[2]/r;
	// for test
	//    d=getAngle(xaxis,yaxis);
	//    TRACE("\nlocal %2d %2d %2d %f ",i,j,k,d);
	//      write(0,*) i,j,k,ang
	// --- step 3 ---->
	//
	d=xaxis[0]*yaxis[1]*zaxis[2]+xaxis[1]*yaxis[2]*zaxis[0]
	+xaxis[2]*yaxis[0]*zaxis[1]-xaxis[2]*yaxis[1]*zaxis[0]
	-xaxis[0]*yaxis[2]*zaxis[1]-xaxis[1]*yaxis[0]*zaxis[2];
	//TRACE("\nlocal %2d %2d %2d %d %f %f %f %f ",i+1,j+1,k+1,iLayer,xaxis[0],yaxis[0],zaxis[0],d);
	return d;
}

//
// ******* calc anisotropic coeffeciante for i,j,k ********
//
void anfct(short int i, short int j, short int k, float v[3]) {
	int locanf;
	float f[3][3], af[3];
	short int m, n;
	float tmp,tmp1;
	float u[3][3]={
		1.,0.,0.,
		0.,1.,0.,
		0.,0.,1.};

		locanf=*(locXCT[k]+NI*j+i);


		for (m=0; m<3; m++) {
			tmp1=*(fibdir[m]+locanf);
			for (n=0; n<3; n++) {
				tmp=*(fibdir[n]+locanf);
				f[m][n]=u[m][n]+rrat1*tmp1*tmp; //corrected by zhu
			}
		}
		for (m=0; m<3; m++) {
			af[m]=f[m][0]*v[0]+f[m][1]*v[1]+f[m][2]*v[2];
		}
		for (m=0; m<3; m++) {
			v[m]=af[m];
		}
}

// Read the matrix data of the body (a344.data)
void rdmtx(void) {

	short int i;
	HFILE hFp;      
	//short int index;
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'m');
	//filepath.SetAt(index+2,'t');
	//filepath.SetAt(index+3,'x');

	//hFp = _lopen(filepath,OF_READ); 
	hFp=_lopen(dataPath+"tour.mtx ",OF_READ);
	if (hFp==HFILE_ERROR) 
	{
		fprintf(stdout,"can not create the file--mtx\n");
		fflush(stdout);
		flag_flop=1;
		return;
	}


	for (i=0;i<NL;i++) 
		_lread(hFp,aw[i],NL*4);
	_lread(hFp,bw,NL*4);
	_lread(hFp,&alp,4);
	_lclose(hFp);

}

// Make stiml data for inverse excitation
void stminvx(short int ivolpkj) {
	short int ks, nspt, mk, i, j, k;

	//idist=20*ND;
	idist=ivolpkj;
	ks=kVtr+idist;

	for (k=kVtr; k<=kBtm; k++) {
		for (i=0; i<NI; i++) {
			for (j=0; j<NJ; j++) {
				if (*(mapCell[k]+j*NI+i)>4) 
					*(mapAPD[k]+j*NI+i)=7;
				else
					*(mapAPD[k]+j*NI+i)=0;
			}
		}
	}

	mNub=0;

	for (k=kVtr; k<NK; k++) {
		//  back view 
		for (j=0;j<NJ;j++) {
			for (i=0;i<NI;i++) {
				if (*(mapAPD[k]+j*NI+i)==6) {
					break;
				} 
				if (*(mapAPD[k]+j*NI+i)==7) {
					*(mapAPD[k]+j*NI+i)=6;
					if (k<=ks) {
						*(mag[0]+mNub)=i;
						*(mag[1]+mNub)=j;
						*(mag[2]+mNub)=k;
						*(mag[3]+mNub)=0; 
						mNub++;
					}
					break;
				}
			} 
		}

		// front view
		for (j=0; j<NJ; j++) {
			for (i=NI-1; i>-1; i--) {
				if (*(mapAPD[k]+j*NI+i)==6) { 
					break;
				} 
				if (*(mapAPD[k]+j*NI+i)==7 ) {
					*(mapAPD[k]+j*NI+i)=6;
					if (k<=ks) {
						*(mag[0]+mNub)=i;
						*(mag[1]+mNub)=j;
						*(mag[2]+mNub)=k;
						*(mag[3]+mNub)=0;
						mNub++;
					}
					break;
				}
			}
		}  

		// right view
		for (i=0;i<NI;i++) {
			for (j=0;j<NJ;j++) {
				if (*(mapAPD[k]+j*NI+i)==6) {
					break;
				} else if (*(mapAPD[k]+j*NI+i)==7) {
					*(mapAPD[k]+j*NI+i)=6;
					if (k<=ks) {
						*(mag[0]+mNub)=i;
						*(mag[1]+mNub)=j;
						*(mag[2]+mNub)=k;
						*(mag[3]+mNub)=0;
						mNub++;
					}
					break;
				}
			}
		} 

		// left view
		for (i=0;i<NI;i++) {
			for (j=NJ-1;j>-1;j--) {
				if (*(mapAPD[k]+j*NI+i)==6) { 
					break;
				}  
				if (*(mapAPD[k]+j*NI+i)==7 ) {
					*(mapAPD[k]+j*NI+i)=6;
					if (k<=ks) {
						*(mag[0]+mNub)=i;
						*(mag[1]+mNub)=j;
						*(mag[2]+mNub)=k;
						*(mag[3]+mNub)=0;
						mNub++;
					}
					break;
				}
			}
		} 
	}

	// the most low layer 
	for (i=0;i<NI;i++) {
		for (j=0;j<NJ;j++) { 
			if (*(mapAPD[kBtm]+j*NI+i)==7) { 
				*(mapAPD[kBtm]+j*NI+i)=6;
				if (kBtm<=ks) {
					*(mag[0]+mNub)=i;
					*(mag[1]+mNub)=j;
					*(mag[2]+mNub)=kBtm;
					*(mag[3]+mNub)=0;
					mNub++;
				}
			}
		}
	}

	// septum setting
	nspt=0;
	for (k=kVtr;k<=kBtm;k++) {
		for (i=0;i<NI;i++) {
			mk=0;
			for (j=1;j<NJ;j++) {
				if ((*(mapAPD[k]+(j-1)*NI+i)==0)&&(*(mapAPD[k]+j*NI+i)==7)&&(mk==1)) {
					nspt=nspt+1;  
					*(mapAPD[k]+j*NI+i)=6;
					break;
				}
				if ((*(mapAPD[k]+j*NI+i)==7)&&(*(mapAPD[k]+(j+1)*NI+i)==0))   
					mk=1;
			}	
		} 
	}
	// testing
	/*
	TRACE("\n35\n");
	for (j=0;j<NJ;j++) { 
	for (i=0;i<NI;i++) {
	TRACE("%d",*(mapAPD[34]+j*NI+i));
	}
	TRACE("\n");
	}
	TRACE("\n40\n");
	for (j=0;j<NJ;j++) { 
	for (i=0;i<NI;i++) {
	TRACE("%d",*(mapAPD[39]+j*NI+i));
	}
	TRACE("\n");
	}
	*/

} 

// APD distribution 
void XCTinvcm(void) {
	short int * iACTv[3];
	short int jACTv[3][NI*ND],kACTv[3][NI*ND];
	short int idir[12];
	short int iseqx[12]={-1,-1, 0, 0, 1, 0, 1, 1, 0, 0,-1, 0 };
	short int iseqy[12]={ 0, 1, 1, 0, 0, 1, 0,-1,-1, 0, 0,-1 };
	short int iseqz[12]={ 0, 0, 0,-1,-1,-1, 0, 0, 0, 1, 1, 1 };
	short int ix,iy,iz,jx,jy,jz,l;
	short int jdist,jx0,jy0,jz0,mappu,mappu0;
	long i,j,k,nACTv,mACTv,ncont;
	long nblck,nStep,nbrch;
	//  unsigned char mappu,mappu0;
	//idist=20*ND;
	//------ initialize mapACT ---------
	for(i=0;i<3;i++) {
		iACTv[i]=(short int *) malloc(50000*ND3*2);
		if(iACTv[i]==NULL) {
			MessageBox(NULL,"Out of memory !",NULL,MB_OK);
			flag_flop=1;
			return;
		}
	}
	for(k=0;k<NK;k++) {
		for(j=0;j<NJ;j++) {
			for(i=0;i<NI;i++) {
				*(mapACT[k]+j*NI+i)=0;
			}
		}
	}
	for(i=0;i<3;i++) {
		for(j=0;j<50000*ND3;j++) {
			*(iACTv[i]+j)=0;
		}
	}
	nblck=0;
	ic=0;
	nACTv=0;

	// mapAPD[]: a map contains value = 6 (boundary) and value = 7 (ventricular)

	while (1) { 
		// TRACE("\nmNub = %d",mNub);
		for(i=0;i<mNub;i++) { //for example, mNub=12322
			if (*(mag[3]+i)!=ic) continue;
			jx=*(mag[0]+i);
			jy=*(mag[1]+i);
			jz=*(mag[2]+i);
			mappu=*(mapAPD[jz]+jy*NI+jx);
			//        nACTv=nACTv+1;
			*(iACTv[0]+nACTv)=jx;
			*(iACTv[1]+nACTv)=jy;
			*(iACTv[2]+nACTv)=jz;
			*(mapACT[jz]+jy*NI+jx)=ic;
			//        *(mapAPD[jz]+jy*NI+jx)=mappu+20;
			*(mapAPD[jz]+jy*NI+jx)=mappu+20*ND;
			nACTv++;
		}
		ic=ic+1;
		// TRACE("\nnACTv= %2d %5d ", ic, nACTv);
		nACTv=0;
		for(k=kVtr;k<=kBtm;k++) {
			for(i=0;i<NI;i++) {
				for(j=0;j<NJ;j++) {
					mappu=*(mapAPD[k]+j*NI+i);
					if((mappu<6)||(mappu>7)) continue; // exculde 0 and others, if any
					ncont=0;
					for(l=0;l<12;l++) {
						idir[l]=0;
						ix=i+iseqx[l];
						if((ix<0)||(ix>(NI-1))) continue;
						iy=j+iseqy[l];
						if((iy<0)||(iy>(NJ-1))) continue;
						iz=k+iseqz[l];
						if((iz<kVtr)||(iz>kBtm)) continue;
						mappu0=*(mapAPD[iz]+iy*NI+ix);
						if((mappu0<20*ND+6)||(mappu0>20*ND+7)) continue;
						ncont=ncont+1;
					}
					if(ncont==0) continue;
					//      if((mappu==6)||(mappu==7)) mappu=mappu+10;
					//      *(mapAPD[k]+j*NI+i)=mappu;
					if((mappu==6)||(mappu==7)) *(mapAPD[k]+j*NI+i)+=10*ND;
					//      nACTv=nACTv+1;
					*(iACTv[0]+nACTv)=i;
					*(iACTv[1]+nACTv)=j;
					*(iACTv[2]+nACTv)=k;
					*(mapACT[k]+j*NI+i)=ic;
					nACTv++;
					//if(k==120 && *(mapACT[k]+j*NI+i) > 0) {
					//	TRACE(" %d ",*(mapACT[k]+j*NI+i));
					//}
				}
			}

		}
		// Conductive system
		mACTv=nACTv;
		for (i=0;i<nACTv;i++) {
			jx=*(iACTv[0]+i);
			jy=*(iACTv[1]+i);
			jz=*(iACTv[2]+i);
			mappu=*(mapAPD[jz]+jy*NI+jx);

			if (mappu != 10*ND+6) 
				continue;

			jACTv[0][0]=jx;
			jACTv[1][0]=jy;
			jACTv[2][0]=jz;
			nStep=0;
			nbrch=1;
			jdist=1;

			while (1) {
				for (j=0;j<nbrch;j++) {
					jx0=jACTv[0][j];
					jy0=jACTv[1][j];
					jz0=jACTv[2][j];

					for (l=0;l<12;l++) {
						jx=jx0+iseqx[l];
						if ((jx<=-1)||(jx>NI-1)) continue; // <0
						jy=jy0+iseqy[l];
						if ((jy<=-1)||(jy>NJ-1)) continue; // <0 
						jz=jz0+iseqz[l];
						if ((jz<kVtr)||(jz>kBtm)) continue;
						mappu=*(mapAPD[jz]+jy*NI+jx);
						if (mappu!=6) continue;
						kACTv[0][nStep]=jx;
						kACTv[1][nStep]=jy;
						kACTv[2][nStep]=jz;
						nStep++;
						*(iACTv[0]+mACTv)=jx;
						*(iACTv[1]+mACTv)=jy;
						*(iACTv[2]+mACTv)=jz;
						*(mapACT[jz]+jy*NI+jx)=ic;   
						*(mapAPD[jz]+jy*NI+jx)=mappu+10*ND;
						mACTv++;
					}
				}
				if (nStep==0) break;
				jdist=jdist+1;
				for (k=0;k<nStep;k++) {
					jACTv[0][k]=kACTv[0][k];
					jACTv[1][k]=kACTv[1][k];
					jACTv[2][k]=kACTv[2][k];
				}
				if (jdist>=idist) break;
				nbrch=nStep;
				nStep=0;
			}     
		}

		nACTv=mACTv;
		// The next circle
		for (i=0;i<NI;i++) 
			for (j=0;j<NJ;j++) 
				for (k=kVtr;k<=kBtm;k++) {
					mappu=*(mapAPD[k]+j*NI+i);
					if ((mappu>30*ND+7)||(mappu<10*ND+6)) continue;
					*(mapAPD[k]+j*NI+i)=mappu+10*ND;
				}
				if((nblck!=0)&&(nACTv==0)) break;
				nblck=nblck+nACTv;
	}			
	maxlay=ic+1;
	// Display
	// Steps were needed to compute excitation
	// ventricular processes are completed.
	// total excited units = nblck
	// TRACE("\n%d  steps were needed to compute excitation, total excited units = %d ", ic, nblck);

	/*
	TRACE("\nk= 35\n");

	for (i=0; i<NI; i++) {
	TRACE("i= %d\n",i);
	for (j=0; j<NJ; j++) {
	TRACE("%2d",*(mapACT[35-1]+j*NI+i));
	}
	}
	TRACE("\nk= 40\n");
	for (i=0; i<NI; i++) {
	TRACE("i= %d\n",i);
	for (j=0; j<NJ; j++) {
	TRACE("%2d",*(mapACT[40-1]+j*NI+i));
	}
	}

	*/
	for (i=0;i<3;i++) {
		free(iACTv[i]);
	}

}

//  mapACT <-- deference of Phase 2 from defined value (ms)
void savACT(void) {

	char iCell;
	short int i,j,k,m;
	int idev,init,md;
	HFILE hFp;
	for (k=0;k<NK;k++) 
		for (i=0;i<NI;i++) 
			for (j=0;j<NJ;j++) {
				if (*(mapACT[k]+j*NI+i)<1) {
					*(mapAPD[k]+j*NI+i)=*(mapACT[k]+j*NI+i);
					continue;
				}
				iCell=*(mapCell[k]+j*NI+i);
				if (iCell==15) {
					*(mapAPD[k]+j*NI+i)=*(mapACT[k]+j*NI+i);
					continue;
				}
				//*(mapACT[k]+j*NI+i)*= *(iparm+(iCell-1)*NPARM+9);  
				*(mapAPD[k]+j*NI+i)=*(mapACT[k]+j*NI+i)*(*(iparm+(iCell-1)*NPARM+9));  
			}

			// Random distribution of the APD
			for (iCell=1;iCell<=NCELL;iCell++) {
				if (*(iparm+(iCell-1)*NPARM+11)<=0) continue;
				idev=*(iparm+(iCell-1)*NPARM+11)*(*(iparm+(iCell-1)*NPARM+2));
				init=idev;
				for (k=0;k<NK;k++) {
					for (i=0;i<NI;i++) {
						for (j=0;j<NJ;j++) {
							if (*(mapCell[k]+j*NI+i)!=iCell) 
								continue;
							init=init*65+1;
							md=init%256;
							*(mapAPD[k]+j*NI+i) = (short int)(idev*(md-128)/12800);
							init=md;
						}
					}
				}
			}

			// Save file of ACT 
			CFile f;
			CFileException e;
			//short int index;

			//index=filepath.FindOneOf(".");
			//filepath.SetAt(index+1,'a');
			//filepath.SetAt(index+2,'c');
			//filepath.SetAt(index+3,'t');

			//hFp=_lcreat(dataPath+"tour.act ",0);
			//if (hFp==HFILE_ERROR) 
			//{
			//	 fprintf(stdout,"can not create the file--act\n");
			//	fflush(stdout);
			//	return;
			//}

			if (!f.Open( dataPath+"tour.act ", CFile::modeCreate | CFile::modeWrite, &e )) {
#ifdef _DEBUG
				afxDump << "File could not be opened " << e.m_cause << "\n";
#endif
			}


			//	if (!f.Open( filepath, CFile::modeCreate | CFile::modeWrite, &e )) {
			//#ifdef _DEBUG
			//		afxDump << "File could not be opened " << e.m_cause << "\n";
			//#endif
			//	}
			f.Write(kmin,2*NI*NJ);
			f.Write(kmax,2*NI*NJ);
			f.Write(&ic,2);
			for (i=0;i<NI;i++) {
				for (j=0;j<NJ;j++) {
					if (*(kmin+j*NI+i)==NK+1) 
						continue;
					for (k=*(kmin+j*NI+i);k<=*(kmax+j*NI+i);k++)
						m=*(mapAPD[k]+j*NI+i)/6/ND;

					f.Write(&m,2); // hui modify from 1 to 2
				}
			}
			f.Close();
}

//  mapACT <-- deference of Phase 2 from defined value (ms)
/*
void savACT(int myid) {

char iCell;
short int i,j,k,m;
int idev,init,md;
HFILE hFp;
for (k=0;k<NK;k++) 
for (i=0;i<NI;i++) 
for (j=0;j<NJ;j++) {
if (*(mapACT[k]+j*NI+i)<1) {
*(mapAPD[k]+j*NI+i)=*(mapACT[k]+j*NI+i);
continue;
}
iCell=*(mapCell[k]+j*NI+i);
if (iCell==15) {
*(mapAPD[k]+j*NI+i)=*(mapACT[k]+j*NI+i);
continue;
}
//*(mapACT[k]+j*NI+i)*= *(iparm+(iCell-1)*NPARM+9);  
*(mapAPD[k]+j*NI+i)=*(mapACT[k]+j*NI+i)*(*(iparm+(iCell-1)*NPARM+9));  
}

// Random distribution of the APD
for (iCell=1;iCell<=NCELL;iCell++) {
if (*(iparm+(iCell-1)*NPARM+11)<=0) continue;
idev=*(iparm+(iCell-1)*NPARM+11)*(*(iparm+(iCell-1)*NPARM+2));
init=idev;
for (k=0;k<NK;k++) {
for (i=0;i<NI;i++) {
for (j=0;j<NJ;j++) {
if (*(mapCell[k]+j*NI+i)!=iCell) 
continue;
init=init*65+1;
md=init%256;
*(mapAPD[k]+j*NI+i) = (short int)(idev*(md-128)/12800);
init=md;
}
}
}
}

// Save file of ACT 
CFile f;
CFileException e;
//short int index;

//index=filepath.FindOneOf(".");
//filepath.SetAt(index+1,'a');
//filepath.SetAt(index+2,'c');
//filepath.SetAt(index+3,'t');

//hFp=_lcreat(dataPath+"tour.act ",0);
//if (hFp==HFILE_ERROR) 
//{
//	 fprintf(stdout,"can not create the file--act\n");
//	fflush(stdout);
//	return;
//}
if (myid==0){
if (!f.Open( dataPath+"tour.act ", CFile::modeCreate | CFile::modeWrite, &e )) {
#ifdef _DEBUG
afxDump << "File could not be opened " << e.m_cause << "\n";
#endif
}


//	if (!f.Open( filepath, CFile::modeCreate | CFile::modeWrite, &e )) {
//#ifdef _DEBUG
//		afxDump << "File could not be opened " << e.m_cause << "\n";
//#endif
//	}
f.Write(kmin,2*NI*NJ);
f.Write(kmax,2*NI*NJ);
f.Write(&ic,2);
};
for (i=0;i<NI;i++) {
for (j=0;j<NJ;j++) {
if (*(kmin+j*NI+i)==NK+1) 
continue;
for (k=*(kmin+j*NI+i);k<=*(kmax+j*NI+i);k++)
m=*(mapAPD[k]+j*NI+i)/6/ND;

if (myid==0)	f.Write(&m,2); // hui modify from 1 to 2
}
}
if (myid==0)	f.Close();
}
*/
// **************** sub excitation ********************
void XCTcalm(int myid) {
	//      FILE *fp;
	void wtXCTm(short int,short int,short int,short int);
	void bbDLYm(short int,short int,short int);
	void rdXCTm(short int,short int,short int,short int);

	short int itmp, tmp;
	short int iStm,ires,irp,irel,ist,kBB;
	float phsft,mxDLY,mACCl,icross,delt;
	char mCell,iCell,kCell;
	short int *iACTv[4];
	short int *iACTvOld[4];
	short int *jACTv[4];
	short int *kACTv[4];
	short int *iXCT[NK];
	short int *iXCTapd[NK];
	short int *iXCTOld[NK];
	short int iseqx[12]={-1,-1, 0, 0, 1, 0, 1, 1, 0, 0,-1, 0};
	short int iseqy[12]={ 0, 1, 1, 0, 0, 1, 0,-1,-1, 0, 0,-1};
	short int iseqz[12]={ 0, 0, 0,-1,-1,-1, 0, 0, 0, 1, 1, 1};
	short int ix,iy,iz,jx,jy,jz,iv,l;
	short int jdist,jx0,jy0,jz0,is,ICL,ivel;
	short int iSTOP, iS1S2, dS1S2Old, iCell5Ex;
	long i,j,k,nACTv,mACTv,nACTvOld;
	long nblck,nStep,nbrch;
	// >>>>>>> aniso >>>>>>
	float xani,yani,zani,dani,elp;
	float dxani,dyani,dzani;
	short int itms1=0;
	// ---- for vtr aniso use 
	// storing the ellipsoid propagation times ---
	//--------- maximum excitation time Step: maxXctStep -------------
	for(i=0;i<4;i++) {
		iACTv[i]	= (short int *) malloc(50000*ND3*2);
		iACTvOld[i]	= (short int *) malloc(50000*ND3*2);
		jACTv[i]	= (short int *) malloc(50000*ND3*2);
		kACTv[i]	= (short int *) malloc(50000*ND3*2);
		if((iACTv[i]==NULL)||(iACTvOld[i]==NULL)||
			(jACTv[i]==NULL)||(kACTv[i]==NULL)) {
				MessageBox(NULL,"Out of memory !",NULL,MB_OK);
				return;
		}
	}      
	for(i=0;i<NK;i++) {
		iXCT[i]	= (short int *) malloc(NI*NJ*2);
		iXCTapd[i]	= (short int *) malloc(NI*NJ*2);
		iXCTOld[i] = (short int *) malloc(NI*NJ*2);
		if((iXCT[i]==NULL)||(iXCTOld[i]==NULL)) {
			MessageBox(NULL,"Out of memory !",NULL,MB_OK);
			return;
		}
	}
	for(i=0;i<4;i++) {
		for(j=0;j<50000*ND3;j++) {
			*(iACTv[i]+j)=0;
			*(iACTvOld[i]+j)=0;
			*(jACTv[i]+j)=0;
			*(kACTv[i]+j)=0;
		}
	}    
	// --- file mapXCT is initialized with INFTIME ----
	for(i=0;i<NCYCL;i++) {
		for(j=0;j<50000*ND3;j++) {
			*(mapXCTm[i]+j)=INFTIME;
		}
	}
	for(k=0;k<NK;k++) {
		for(j=0;j<NJ;j++) {
			for(i=0;i<NI;i++) {
				*(iXCT[k]+j*NI+i)=INFTIME;
				*(iXCTapd[k]+j*NI+i)=0;
				*(iXCTOld[k]+j*NI+i)=INFTIME;
			}
		}
	}
	mxcycle=0;
	short int tested[NCELL];

	for(i=0;i<NCELL;i++)
		tested[i]=0;
	for(i=0;i<nttl;i++) {
		jx=ipttl[0][i]; /*<Comment by ALF> pos of ith cell*/
		jy=ipttl[1][i];
		jz=ipttl[2][i];
		iCell=*(mapCell[jz]+jy*NI+jx); /*<Comment by ALF> cell type index */
		if(tested[iCell-1]==0)
		{*(iparm+(iCell-1)*NPARM+18)+=ipttl[3][i];tested[iCell-1]=1;
		if (iCell!=1) {*(iparm+(1-1)*NPARM+18)+=ipttl[3][i];//maxXctStep+=ipttl[3][i];
		}
		}

		//TRACE("\nNTTL (%3d %3d %3d) %2d",jx,jy,jz,iCell);
		// set pacemake time of no. 5 cells  
		if (iCell==5) {
			ipstm[0][i]=100*ND/(ipttl[3][i]+1);
			if((ipstm[0][i]*ipttl[3][i])<100*ND) ipstm[0][i]+=1;
			//ipstm[0][i]=100/(ipttl[3][i]+1);
			//if((ipstm[0][i]*ipttl[3][i])<100) ipstm[0][i]+=1;
			//TRACE("\nCell 5, (%d %d %d) %d %d",jx,jy,jz, ipttl[3][i],ipstm[0][i]);
			continue;
		}
		// iparm(n,18) = BCL  basic cycle length (ms) of pacing
		// iparm(n,20) = inc  increament of BCL(ms/cycle)
		ipstm[0][i]=*(iparm+(iCell-1)*NPARM+17);
		ipstm[1][i]=*(iparm+(iCell-1)*NPARM+19);
		ipstm[2][i]=0;
	}   
	nblck=0;
	ic=0;
	nACTv=0;
	iS1S2=0;
	iCell5Ex=0;

	// ------ stimulus: pacemaker spontanous firing -------	
	while (1) {
		// In this loop, ipttl[3][i] is mainly used	to
		// decide ipstm[0][i] and itself	
		jx=0;
		jy=0;
		jz=0;
		iStm=0;
		excited=0;
		for (i=0;i<nttl;i++) {
			jx=ipttl[0][i];
			jy=ipttl[1][i];
			jz=ipttl[2][i];
			iStm=ipttl[3][i];
			iCell=*(mapCell[jz]+jy*NI+jx);

			//TRACE("\nStimulus (%3d %3d %3d)%2d %d %d",jx,jy,jz,iCell,iStm, mxcycle);
			//TRACE("\nbreak1 mxcycle=%d NCYCL=%d ic=%d iCell=%d, iStm=%d, mS2BN=%d,ipstm=%d",mxcycle, NCYCL,ic, iCell, iStm,*(iparm+(iCell-1)*NPARM+18),ipstm[0][i]);
			if (iCell==5) continue; // ignore BB
			if (iStm != ic) continue;
			// ic: i-th time Step
			// nACTv: number of exitation cells at ic time but cellType != 5 (BB)
			// --- end ---
			//TRACE("\nbreak1 mxcycle=%d NCYCL=%d ic=%d iCell=%d, iStm=%d",mxcycle, NCYCL,ic, iCell, iStm);
			nACTv=nACTv+1;
			*(iACTv[0]+nACTv)=jx;
			*(iACTv[1]+nACTv)=jy;
			*(iACTv[2]+nACTv)=jz;
			*(iACTv[3]+nACTv)=*(iparm+(iCell-1)*NPARM+31);  /*<Comment by ALF> iparm store each cell's parameters*/
			// iparm(n,32): conduction speed 
			wtXCTm(ic,jx,jy,jz);
			//if (jx==101 && jy==77 && jz==6) TRACE("\nA mxcycle=%d at ic=%d, iCell=%d",mxcycle,ic,iCell);
			//if (iCell <3) TRACE("\nA %d %d %d %d %d %d",iCell,jx,jy,jz,ic,nACTv);
			// write to file

			// mxcycle: maximum cycle

			if(mxcycle>=NCYCL) {
				break;
			}
			// --- store current time to iXCT and last time to iXCTOld -->
			*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx); // init is INFTIME
			*(iXCT[jz]+jy*NI+jx)=ic;
			excited=1;

			// Update ipttl[3][i]
			// iparm(n,18) = BCL: basic cycle length (ms) of pacing
			// Normally, only SN has this parameter > 0
			/*if(*(iparm+(iCell-1)*NPARM+17)>0) {
			if ((iS1S2==1) && (mS2BN>1)) {
			itmp=ipttl[3][i]+mS2CL;
			mS2BN--;
			} else {
			itmp=ipttl[3][i]+ipstm[0][i];
			}
			dS1S2Old=ipstm[2][i];
			ipstm[0][i] = ipstm[0][i] + ipstm[1][i];
			iCell5Ex=0;
			// ipstm[1][i] is the step
			// iparm(n,19) = pBN: beat number
			// judge by ipttl[3][i]
			if(itmp>*(iparm+(iCell-1)*NPARM+18)) continue;

			if ((mS2ST/3 > ipttl[3][i]) &&(mS2ST/3 < itmp)) {
			ipttl[3][i]=(short int)(mS2ST/3);
			iS1S2=1;
			} else {
			ipttl[3][i]=itmp;
			}	
			ipstm[2][i]=itmp-ipttl[3][i];
			//TRACE("\nTime=%d, %d, %d, %d, %d %d",ic,itmp,ipttl[3][i],dS1S2Old, ipstm[0][i],ipstm[1][i]);      
			continue;
			}*/

			if(*(iparm+(iCell-1)*NPARM+17)>0) {
				if (iCell==1) {
					itmp=ipttl[3][i]+ipstm[0][i];
					dS1S2Old=ipstm[2][i];
					ipstm[0][i] = ipstm[0][i] + ipstm[1][i];
					iCell5Ex=0;
					if(itmp>*(iparm+(iCell-1)*NPARM+18)) continue;
					ipttl[3][i]=itmp; continue;}
				else
				{

					itmp=ipttl[3][i]+ipstm[0][i];
					dS1S2Old=ipstm[2][i];
					ipstm[0][i] = ipstm[0][i] + ipstm[1][i];
					iCell5Ex=0;
					if(itmp>*(iparm+(iCell-1)*NPARM+18)-ipstm[0][i]+3) continue;
					ipttl[3][i]=itmp;

				}
				continue;
			}

			// iparm(n,24) = ICL: intrinsic cycle length(ms)
			ipttl[3][i] = ipttl[3][i] + *(iparm+(iCell-1)*NPARM+23); 
		}

		// ---- display the excitation number ----		
		// go to next Step
		nblck = nblck + nACTv;
		//TRACE("\nmxcycle =%d Step=%3d, number=%ld nblck=%ld ",mxcycle,ic,nACTv, nblck);
		ic = ic + 1;
		//TRACE("\nbreak2 ic=%d maxXctStep=%d ",ic, maxXctStep);
		if (ic>=maxXctStep) break;
		if (nACTv == 0) continue;

		/**
		* very important
		*/
		// --------- propagation (2000)------------>
		nACTvOld=0;
		// nACTv: at moment t, the number of excited cells 
		for (i=1;i<=nACTv;i++) {
			excited=1;
			ix=*(iACTv[0]+i);
			iy=*(iACTv[1]+i);
			iz=*(iACTv[2]+i);
			iv=*(iACTv[3]+i);
			iCell=*(mapCell[iz]+iy*NI+ix);
			//if (ix == 64 && iy == 50 && iz == 64) TRACE("\nB AVN %d",iCell);

			//----------- low conduction speed part ----------->
			// iparm(n,32): conduction speed 
			if (*(iparm+(iCell-1)*NPARM+31)<=0) continue; 
			if (iCell==5) iCell5Ex=1;
			//if (iCell==8) TRACE("\nCell=8  %d, %d, %d, ic=%d, %d %d",ix,iy,iz,ic,iv,mBCL);
			// 100 = Conduction Speed of ATR?
			if (iv<100) {
				nACTvOld=nACTvOld+1;
				*(iACTvOld[0]+nACTvOld)=ix;
				*(iACTvOld[1]+nACTvOld)=iy;
				*(iACTvOld[2]+nACTvOld)=iz;
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)+*(mapSpeed[iz]+iy*NI+ix); //added by zhu
				if (iCell==5) {
					/*ibbDLY=0;
					bbDLYm(ix,iy,iz);
					if (ibbDLY>0)
					*(iACTvOld[3]+nACTvOld)=iv+ibbDLY;
					TRACE("\nBB, %d",*(iACTvOld[3]+nACTvOld));            
					*/

					ibbDLY=0;
					// Add for BB interval by hui wang
					ibbSTEP=0;
					bbDLYm(ix,iy,iz);

					// End of add for BB interval by hui wang, modified by zhu


					if (ibbDLY>0) {ibbSTEP+=nbbSTEP;ibbDLY=100*ND/(ibbSTEP+1);}

					if(ibbDLY>0 && (ibbDLY*ibbSTEP)<100*ND) ibbDLY+=1;

					if (ibbDLY>0)
						*(iACTvOld[3]+nACTvOld)=iv+ibbDLY;
					else
						*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31);

					continue;
				}
				/*if (iCell==3 || iCell==6) {
				if (*(iXCTOld[iz]+iy*NI+ix)==INFTIME)
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31);
				else {

				irel = *(iXCT[iz]+iy*NI+ix)-*(iXCTOld[iz]+iy*NI+ix)-(*(iparm+NPARM*(iCell-1)+4)+*(mapAPD[iz]+iy*NI+ix))/3;

				//irel = *(iXCT[iz]+iy*NI+ix)-*(iXCTOld[iz]+iy*NI+ix)-(*(iparm+NPARM*(iCell-1)+4))/3;
				irel = 3*irel;

				if (irel<*(iparm+NPARM*(iCell-1)+5)) {
				tmp=100+*(iparm+NPARM*(iCell-1)+32)
				-irel*(*(iparm+NPARM*(iCell-1)+32))/(*(iparm+NPARM*(iCell-1)+5));
				if (tmp!=0) {
				ivel = 100*(*(iparm+NPARM*(iCell-1)+31))/tmp;
				} else {
				ivel=*(iparm+NPARM*(iCell-1)+31);
				}
				} else {  
				//  <--- time of RRP stored in iparm(6) ---
				ivel=*(iparm+NPARM*(iCell-1)+31);
				}

				*(iACTvOld[3]+nACTvOld)=iv+ivel;}

				}*/

				/*else if (iCell==3) {
				if (iCell5Ex==0) {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)- dS1S2Old/20;
				TRACE("\nCell=3 E dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else {
				if (mBCL<600&&dS1S2Old<140/3) {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)- (dS1S2Old+67)/33;						
				TRACE("\nCell=3 A dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else if (mBCL<600&&dS1S2Old>=140/3) {
				*(iACTvOld[3]+nACTvOld)=iv;
				TRACE("\nCell=3 B dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else if (mBCL>=600&&dS1S2Old<=210/3) {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31);
				TRACE("\nCell=3 C dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)- dS1S2Old/12;
				TRACE("\nCell=3 D dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} 						
				}
				}*/
				continue; 
			}

			// ------- neighbourhood search (2100) -------->
			// go to iv > 100 situation and set ires = the part of iv < 100
			ires=iv-100*(int)(iv/100);

			for (l=0;l<12;l++) {
				jx=ix+iseqx[l];
				if ((jx<=-1)||(jx>(NI-1))) continue; // >=0 <NI
				jy=iy+iseqy[l];
				if ((jy<=-1)||(jy>(NJ-1))) continue; // >=0 <NJ
				jz=iz+iseqz[l];
				if ((jz<=-1)||(jz>(NK-1))) continue; // >=0 <NK
				// >>>>> aniso: within the ellpisoid ? >>>>>>>>>>>>
				if (ANISO==1 && iCell==7) {
					dani=local(ix,iy,iz);
					//TRACE("\nx,y,z,dani, %2d %2d %2d %f",ix,iy,iz,dani);
					// -- if can't solve local coordinates, treat as isotropic -->
					if (dani > 0.0001) {
						//lctran(iseqx[l],iseqy[l],iseqz[l],dani,xani,yani,zani);
						xani=iseqx[l]*tmswf[0][0]+iseqy[l]*tmswf[0][1]+iseqz[l]*tmswf[0][2];
						yani=iseqx[l]*tmswf[1][0]+iseqy[l]*tmswf[1][1]+iseqz[l]*tmswf[1][2];
						zani=iseqx[l]*tmswf[2][0]+iseqy[l]*tmswf[2][1]+iseqz[l]*tmswf[2][2];
						dxani=xani*yaxis[1]*zaxis[2]+yani*yaxis[2]*zaxis[0]
						+zani*yaxis[0]*zaxis[1]-zani*yaxis[1]*zaxis[0]
						-xani*yaxis[2]*zaxis[1]-yani*yaxis[0]*zaxis[2];

						dyani=xaxis[0]*yani*zaxis[2]+xaxis[1]*zani*zaxis[0]
						+xaxis[2]*xani*zaxis[1]-xaxis[2]*yani*zaxis[0]
						-xaxis[0]*zani*zaxis[1]-xaxis[1]*xani*zaxis[2];

						dzani=xaxis[0]*yaxis[1]*zani+xaxis[1]*yaxis[2]*xani
							+xaxis[2]*yaxis[0]*yani-xaxis[2]*yaxis[1]*xani
							-xaxis[0]*yaxis[2]*yani-xaxis[1]*yaxis[0]*zani;
						xani=dxani/dani;
						yani=dyani/dani;
						zani=dzani/dani;	
						// itms=maps(ix,iy,iz)+1
						//TRACE("\nd  %f %f %f %f",dxani,xani,yaxis[0],zaxis[0]);
						itms1=*(iXCTapd[iz]+iy*NI+ix);    								
						elp=xani*xani/vt2[itms1]+
							yani*yani/vt2[itms1]+
							zani*zani/vl2[itms1];
						// write(0,*) x,y,z,elp
						// TRACE("\n %d %f",itms1,elp);
						if (elp > 1.0) continue;
					}
				}
				// <<<<<<<<<<<<<<<<<<<< aniso <<<<<<<<<<<<<<<<<<<
				mCell=*(mapCell[jz]+jy*NI+jx);
				if ((iCell<=7)&&(mCell<=7)&&(((iCell-mCell)>1)||
					((iCell-mCell)<-1))) continue;
				if ((*(iparm+NPARM*(mCell-1)+33)>0)&&( (mCell>7 && iCell>7 && mCell!=iCell) || (mCell<=7 && iCell>mCell) || (mCell>7 && !(iCell==mCell || iCell==2)))) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)>0)&&( (mCell<=7 && iCell>mCell) || (mCell>7 && !(iCell==mCell || iCell==2)))) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)>0)&&(iCell>mCell)) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)<0)&&(iCell<mCell)) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)<0)&&( (mCell<=7 && iCell<mCell) || (mCell>7 && !(iCell==mCell || iCell==7)))) continue;
				if ((*(iparm+NPARM*(mCell-1)+33)<0)&&( (mCell>7 && iCell>7 && mCell!=iCell) || (mCell<=7 && iCell<mCell) || (mCell>7 && !(iCell==mCell || iCell==7)))) continue;

				//if (jx == 64 && jy == 50 && jz == 64) TRACE("\nC AVN %d",mCell);
				if (mCell<=0) continue; // continue;
				if (mCell>=15) continue; // continue;
				if (mCell==5) iCell5Ex=1;
				//if (iCell==8) TRACE("\nCell=8  %d, %d, %d, ic=%d, %d %d",jx,jy,jz,ic);;
				// --- coupling interval ------>
				idltt=ic-*(iXCT[jz]+jy*NI+jx);
				if (*(iXCT[jz]+jy*NI+jx)==INFTIME) idltt=INFTIME;

				// --- change in cycle length ------>
				idltc=idltt+*(iXCTOld[jz]+jy*NI+jx)-*(iXCT[jz]+jy*NI+jx);
				if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME) idltc=0;
				if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
					*(iparm+(1-1)*NPARM+23)>0)
					idltc = ic-*(iXCT[jz]+jy*NI+jx)-*(iparm+(1-1)*NPARM+23);


				//      rdXCT(ic,jx,jy,jz);
				// irp = time in phase 2 + mapACT/3 +plateau of potential in phase 3 *idltc/100
				// --- absolute refractory period ------>
				//irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3
				//	+*(iparm+(mCell-1)*NPARM+10)*idltc/100;
				irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3;


				//if (mCell==6)
				//        irp=(*(iparm+NPARM*(mCell-1)+4))/3;
				irel=idltt-irp;


				// ++++++++ in absolute refractory period ? +++++++
				if (irel<=0) continue;
				//if (*(mapAPD[jz]+jy*NI+jx)>20) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>20 && idltc<0) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>30 && idltc<0) idltc = 3*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>30 && idltc>0) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>40 && idltc>0) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>40 && idltc>0) idltc = 2*idltc;
				//if (mCell==3) TRACE("\nCell=3  %d, %d, %d,%d,%d,%d",irp,ic,*(iXCT[jz]+jy*NI+jx),*(mapAPD[jz]+jy*NI+jx),*(iparm+(mCell-1)*NPARM+10),idltc);


				*(mapAPD[jz]+jy*NI+jx) += *(iparm+(mCell-1)*NPARM+10)*idltc*3/100; //added by Zhu

				//if (mCell==3) TRACE("\nCell=3  %d, %d, %d",*(mapAPD[jz]+jy*NI+jx)/3,idltc,irp);

				if (*(iXCT[jz]+jy*NI+jx)==INFTIME && mCell==3) {irel=INFTIME;*(mapAPD[jz]+jy*NI+jx)=0;}





				// --- find automaticity in stimul data ----
				// iparm(n,24), ICL: intrinsic cycle length (ms)
				iSTOP =0;
				if (*(iparm+NPARM*(mCell-1)+23)>0) {  // !=0 August 10, 1996
					// <--- next touch time should be beyound ARP of the cell --

					for (is=0;is<nttl;is++) {
						if (jx!=ipttl[0][is]) continue; 
						if (jy!=ipttl[1][is]) continue; 
						if (jz!=ipttl[2][is]) continue; 
						// --- iparm(23) used for adjusting intermediate change
						// of EP intrinsic cycle length --->
						ICL = *(iparm+NPARM*(mCell-1)+23);
						ist = ic-*(iXCT[jz]+jy*NI+jx);

						// PRT: protection indicator
						// --- no protection ---->
						if (*(iparm+NPARM*(mCell-1)+24)==0) {
							if (ist<=irp) continue; //{iSTOP=1;break;}
							//if (iSTOP==1) 
							ipttl[3][is]=ic+ICL; // ICL/3
							/******************/
							nACTvOld=nACTvOld+1; 
							*(iACTvOld[0]+nACTvOld)=jx;
							*(iACTvOld[1]+nACTvOld)=jy;
							*(iACTvOld[2]+nACTvOld)=jz;
							*(iACTvOld[3]+nACTvOld)=*(iparm+NPARM*(mCell-1)+31)+ires;
							wtXCTm(ic,jx,jy,jz);
							if (mxcycle>=NCYCL) {iSTOP=1;break;}
							//if (ic==*(iXCT[jz]+jy*NI+jx)) continue;
							*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
							*(iXCT[jz]+jy*NI+jx)=ic;//added to by zhu
							irel=0;
							excited=1;
							TRACE("\n %d, %d %d %d %d %d",*(iXCTOld[jz]+jy*NI+jx),*(iXCT[jz]+jy*NI+jx),ic,jx,jy,jz);
							/******************/

							//iSTOP=1;
							continue; //break;  // rewrite condition
						}
						if (idltt==INFTIME) continue;

						//ist = ic-*(iXCT[jz]+jy*NI+jx);
						//            if (ist<=irp)        goto loop21; // August 10, 1996
						if (ist<=irp) continue; //{iSTOP=1;break;}
						phsft =(float)100.*(idltt/ICL);
						mxDLY =(float)*(iparm+NPARM*(mCell-1)+25);
						mACCl =(float)*(iparm+NPARM*(mCell-1)+26);
						if (mxDLY == 0 && mACCl == 0) continue;
						icross=(float)*(iparm+NPARM*(mCell-1)+27);
						if (icross == 0 || icross == 100) continue;
						if (phsft<=icross) 
							delt=phsft*mxDLY/icross;
						else 
							delt=mACCl-(phsft-icross)*mACCl/(100-icross);
						// -- store touch time --->
						// -- modify next stimulating time --->
						ipttl[3][is]=ipttl[3][is]+(int)(ICL*delt/100);

						//TRACE("\ntime=%4d,ixt=%4d,idltt=%4d,icl=%4d,phsft=%4d,intermediate=%4d",
						//	ic, *(iXCT[jz]+jy*NI+jx),idltt,ICL, (int)phsft, ipttl[3][is]); 
						// change value after each touch time 
						// avoiding from successive modification by surrounding cells
						if (ic==*(iXCT[jz]+jy*NI+jx)) continue;
						*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
						*(iXCT[jz]+jy*NI+jx)=ic;
						irel=0;
						excited=1;
						//iSTOP=1;
						continue; //break;  // rewrite condition
					}
				} 
				if (iSTOP==1) continue;
				if (irel==0) continue;


				// +++++ special processing for BB +++++
				if (mCell==5) {


					*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
					*(iXCT[jz]+jy*NI+jx)=ic;

					// Add for BB interval by hui wang modified by Zhu
					// variable ibbSTEP, nbbSTEP are added to store steps by first BB
					// ibbSTEP is a function in bbDLYm(i,j,k)
					nbbSTEP=0;
					ibbDLY=0;
					ibbSTEP=0;
					bbDLYm(jx,jy,jz);
					nbbSTEP=ibbSTEP;
					// end of add for BB interval by hui wang
					//ic+=10; // add by hw, BB interval
					//TRACE("\n nHB = %d, ic= %d",nHB,ic);
					for(kBB=0;kBB<nBB;kBB++) { 
						jx=iBB[0][kBB];
						jy=iBB[1][kBB];
						jz=iBB[2][kBB];
						nACTvOld=nACTvOld+1;
						*(iACTvOld[0]+nACTvOld)=jx;
						*(iACTvOld[1]+nACTvOld)=jy;
						*(iACTvOld[2]+nACTvOld)=jz;
						//*(iACTvOld[3]+nACTvOld)=100;
						ibbDLY=0;
						// Add for BB interval by hui wang,modified by zhu
						ibbSTEP=0;
						bbDLYm(jx,jy,jz);
						ibbSTEP+=nbbSTEP;
						ibbDLY=100*ND/(ibbSTEP+1);
						if((ibbDLY*ibbSTEP)<100*ND) ibbDLY+=1;
						// End of add for BB interval by hui wang
						*(iACTvOld[3]+nACTvOld)=ibbDLY;

						wtXCTm(ic,jx,jy,jz);
						//if (jx==101 && jy==77 && jz==6) TRACE("\nB mxcycle=%d at ic=%d, iCell=%d",mxcycle,ic,mCell);
						//if (mCell >2 && mCell <6) TRACE("\nB %d %d %d %d %d %d",mCell,jx,jy,jz,ic,nACTvOld);
						if (mxcycle>=NCYCL) break;
						*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
						*(iXCT[jz]+jy*NI+jx)=ic;
						excited=1;
					}
					continue;
				} 


				nACTvOld=nACTvOld+1; 
				*(iACTvOld[0]+nACTvOld)=jx;
				*(iACTvOld[1]+nACTvOld)=jy;
				*(iACTvOld[2]+nACTvOld)=jz;
				wtXCTm(ic,jx,jy,jz);
				//TRACE("\nbreak3 mxcycle=%d NCYCL=%d ",mxcycle, NCYCL);
				if (mxcycle>=NCYCL) break;
				*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
				*(iXCT[jz]+jy*NI+jx)=ic;

				irel = 3*irel;
				//if (*(iXCTOld[jz]+jy*NI+jx)!=INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
				//			irel>*(iparm+NPARM*(mCell-1)+5))
				//	*(mapAPD[jz]+jy*NI+jx)=0;

				// time of RRP stored in iparm(6)
				if ((irel)<*(iparm+NPARM*(mCell-1)+5)) {
					tmp=100+*(iparm+NPARM*(mCell-1)+32)
						-irel*(*(iparm+NPARM*(mCell-1)+32))/(*(iparm+NPARM*(mCell-1)+5));
					if (tmp!=0) {
						ivel = 100*(*(iparm+NPARM*(mCell-1)+31))/tmp;
					} else {
						ivel=*(iparm+NPARM*(mCell-1)+31);
					}
				} else {  
					//  <--- time of RRP stored in iparm(6) ---
					ivel=*(iparm+NPARM*(mCell-1)+31);
				}
				*(mapSpeed[jz]+jy*NI+jx)=ivel-*(iparm+NPARM*(mCell-1)+31);//added by Zhu

				// test results
				//TRACE("\nmcell=%4d,ic=%4d,idltt=%4d,idltc=%4d,ivel=%4d",mCell,ic,idltt,idltc,ivel);

				if (iCell!=mCell) {
					if (mCell == 5) {
						bbDLYm(jx,jy,jz);
						*(iACTvOld[3]+nACTvOld)=ibbDLY;
						//TRACE("\n BB2=%d, %d %d (%d %d %d) ic=%d ",*(iACTvOld[3]+nACTvOld),iv,ibbDLY,ix,iy,iz, ic);
						continue;
					}
					*(iACTvOld[3]+nACTvOld)=ivel;
					continue;
				}
				*(iACTvOld[3]+nACTvOld)=ivel+ires;				
			}
			// <------- END of neighbourhood search (2100) -----
			// >>>>>>>> anisotropy >>>>>
			if (ANISO==1 && iCell == 7) {
				// ltrat==2;
				if (*(iXCTapd[iz]+iy*NI+ix) < 2) {
					nACTvOld=nACTvOld+1;
					*(iACTvOld[0]+nACTvOld)=ix;
					*(iACTvOld[1]+nACTvOld)=iy;
					*(iACTvOld[2]+nACTvOld)=iz;
					//*(iACTvOld[3]+nACTvOld)=ires+ 
					//	*(iparm+NPARM*(iCell-1)+31);
					*(iACTvOld[3]+nACTvOld)=ires+ 
						*(iparm+NPARM*(iCell-1)+31)+*(mapSpeed[iz]+iy*NI+ix);
					*(iXCTapd[iz]+iy*NI+ix)+=1;
				} else {
					*(iXCTapd[iz]+iy*NI+ix)=0;
				}
			}
			// <<<<<<<<<<<
		}
		// <------- END of propagation (2000) -----

		//  +++++++++++ for high speed ++++++++
		mACTv=nACTvOld;
		// ------- propagation (1000) -------->
		for(i=1;i<=nACTvOld;i++) {
			idist=(int)(*(iACTvOld[3]+i)/100);       
			if (idist<2) continue; 
			*(jACTv[0]+1)=*(iACTvOld[0]+i);
			*(jACTv[1]+1)=*(iACTvOld[1]+i);
			*(jACTv[2]+1)=*(iACTvOld[2]+i);
			ires=*(iACTvOld[3]+i)-idist*100;
			nStep=0;
			nbrch=1;
			jdist=1;
			while (1) {   
				for (j=1;j<=nbrch;j++) {
					jx0=*(jACTv[0]+j);
					jy0=*(jACTv[1]+j);
					jz0=*(jACTv[2]+j);
					mCell=*(mapCell[jz0]+jy0*NI+jx0);
					if (mCell==5) iCell5Ex=1;

					for (l=0;l<12;l++) {
						jx=jx0+iseqx[l];
						if ((jx<=-1)||(jx>(NI-1))) continue;   // <0 or >=NI
						jy=jy0+iseqy[l];
						if ((jy<=-1)||(jy>(NJ-1))) continue;   // <0 or >=NJ
						jz=jz0+iseqz[l];
						if ((jz<=-1)||(jz>(NK-1))) continue;   // <0 or >=NK
						kCell = *(mapCell[jz]+jy*NI+jx);
						//if (jx == 64 && jy == 50 && jz == 64) TRACE("\nE AVN %d",kCell);

						if (kCell != mCell) continue; 
						//++++++++ in effective refractory period ? +++++++

						// >>>>> aniso: within the ellpisoid ? >>>>>>>>>>>>
						if (ANISO==1 && mCell==7) {
							dani=local(jx0,jy0,jz0);
							//TRACE("\nx,y,z,dani, %2d %2d %2d %f",ix,iy,iz,dani);
							// -- if can't solve local coordinates, treat as isotropic -->
							if (dani > 0.0001) {
								//lctran(iseqx[l],iseqy[l],iseqz[l],dani,xani,yani,zani);
								xani=iseqx[l]*tmswf[0][0]+iseqy[l]*tmswf[0][1]+iseqz[l]*tmswf[0][2];
								yani=iseqx[l]*tmswf[1][0]+iseqy[l]*tmswf[1][1]+iseqz[l]*tmswf[1][2];
								zani=iseqx[l]*tmswf[2][0]+iseqy[l]*tmswf[2][1]+iseqz[l]*tmswf[2][2];
								dxani=xani*yaxis[1]*zaxis[2]+yani*yaxis[2]*zaxis[0]
								+zani*yaxis[0]*zaxis[1]-zani*yaxis[1]*zaxis[0]
								-xani*yaxis[2]*zaxis[1]-yani*yaxis[0]*zaxis[2];

								dyani=xaxis[0]*yani*zaxis[2]+xaxis[1]*zani*zaxis[0]
								+xaxis[2]*xani*zaxis[1]-xaxis[2]*yani*zaxis[0]
								-xaxis[0]*zani*zaxis[1]-xaxis[1]*xani*zaxis[2];

								dzani=xaxis[0]*yaxis[1]*zani+xaxis[1]*yaxis[2]*xani
									+xaxis[2]*yaxis[0]*yani-xaxis[2]*yaxis[1]*xani
									-xaxis[0]*yaxis[2]*yani-xaxis[1]*yaxis[0]*zani;
								xani=dxani/dani;
								yani=dyani/dani;
								zani=dzani/dani;	
								// itms=maps(ix,iy,iz)+1
								//TRACE("\nd  %f %f %f %f",dxani,xani,yaxis[0],zaxis[0]);
								itms1=*(iXCTapd[jz0]+jy0*NI+jx0);    								
								elp=xani*xani/vt2[itms1]+
									yani*yani/vt2[itms1]+
									zani*zani/vl2[itms1];
								// write(0,*) x,y,z,elp
								// TRACE("\n %d %f",itms1,elp);
								if (elp > 1.0) continue;
							}
						}

						idltt=ic-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME) idltt=INFTIME;

						// --- change in cycle length ------>
						idltc=idltt+*(iXCTOld[jz]+jy*NI+jx)-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME) idltc=0;
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
							*(iparm+(1-1)*NPARM+23)>0)
							idltc = ic-*(iXCT[jz]+jy*NI+jx)-*(iparm+(1-1)*NPARM+23);

						//      rdXCT(ic,jx,jy,jz);
						// irp = time in phase 2 + mapACT/3 +plateau of potential in phase 3 *idltc/100
						// --- absolute refractory period ------>
						//irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3
						//	+*(iparm+(mCell-1)*NPARM+10)*idltc/100;
						irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3;


						//if (mCell==6)
						//        irp=(*(iparm+NPARM*(mCell-1)+4))/3;
						irel=idltt-irp;
						if (irel<=0) continue; // continue;
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME && mCell==3) {irel=INFTIME;*(mapAPD[jz]+jy*NI+jx)=0;}
						*(mapAPD[jz]+jy*NI+jx) += *(iparm+(mCell-1)*NPARM+10)*idltc*3/100; //added by Zhu
						/*
						idltt=ic-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME)	idltt=INFTIME;
						idltc=idltt+*(iXCTOld[jz]+jy*NI+jx)-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME) idltc=0;
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
						*(iparm+(1-1)*NPARM+23)>0)
						idltc = ic-*(iXCT[jz]+jy*NI+jx)-*(iparm+(1-1)*NPARM+23);
						//      rdXCT(ic,jx,jy,jz);
						// irp=(*(iparm+NPARM*(kCell-1)+4)+*(mapACT[jz]+jy*NI+jx))/3+
						// 		*(iparm+(kCell-1)*NPARM+10)*idltc/100;
						//irp=(*(iparm+NPARM*(kCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3+
						//	*(iparm+(kCell-1)*NPARM+10)*idltc/100;
						irp=(*(iparm+NPARM*(kCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3;
						*(mapAPD[jz]+jy*NI+jx) += *(iparm+(kCell-1)*NPARM+10)*idltc*3/100; //added by Zhu

						irel=idltt-irp;
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME) irel=INFTIME;
						*/

						irel = 3*irel;
						//if (*(iXCTOld[jz]+jy*NI+jx)!=INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
						//	irel>=*(iparm+NPARM*(kCell-1)+5))
						//    *(mapAPD[jz]+jy*NI+jx)=0;

						if ((irel)<*(iparm+NPARM*(kCell-1)+5)) {
							tmp=100+*(iparm+NPARM*(mCell-1)+32)
								-irel*(*(iparm+NPARM*(mCell-1)+32))/(*(iparm+NPARM*(mCell-1)+5));
							if (tmp!=0) {
								ivel = 100*(*(iparm+NPARM*(mCell-1)+31))/tmp;
							} else {
								ivel=*(iparm+NPARM*(mCell-1)+31);
							}
						} else {
							ivel=*(iparm+NPARM*(kCell-1)+31);
						}

						*(mapSpeed[jz]+jy*NI+jx)=ivel-*(iparm+NPARM*(kCell-1)+31);//added by Zhu


						nStep=nStep+1;
						*(kACTv[0]+nStep)=jx;
						*(kACTv[1]+nStep)=jy;
						*(kACTv[2]+nStep)=jz;
						// nStep++;
						mACTv=mACTv+1;
						*(iACTvOld[0]+mACTv)=jx;
						*(iACTvOld[1]+mACTv)=jy;
						*(iACTvOld[2]+mACTv)=jz;
						*(iACTvOld[3]+mACTv)=ivel+ires;

						// mACTv++;
						// TRACE(" D%d,",mACTv);
						wtXCTm(ic,jx,jy,jz);
						//if (jx==101 && jy==77 && jz==6) TRACE("\nD mxcycle=%d at ic=%d, iCell=%d",mxcycle,ic,kCell);
						//if (kCell >2 && kCell <6) TRACE("\nD %d %d %d %d %d %d",kCell,jx,jy,jz,ic,mACTv);
						//TRACE("\nbreak4 mxcycle=%d NCYCL=%d ",mxcycle, NCYCL);
						if (mxcycle>=NCYCL) {
							//TRACE("\nbreak5 iSTOP=%d mxcycle=%d,NCYCL=%d",iSTOP, mxcycle, NCYCL);
							iSTOP =1;
							break;
						}
						*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
						*(iXCT[jz]+jy*NI+jx)=ic;
						excited=1;


					}

					/*// >>>>>>>> anisotropy >>>>>
					if (ANISO==1 && mCell == 7) {
					// ltrat==3;
					if (*(iXCTapd[jz0]+jy0*NI+jx0) < 3) {
					mACTv=mACTv+1;
					*(iACTvOld[0]+mACTv)=jx0;
					*(iACTvOld[1]+mACTv)=jy0;
					*(iACTvOld[2]+mACTv)=jz0;
					*(iACTvOld[3]+mACTv)=ivel+ires;
					//*(iACTvOld[3]+nACTvOld)=ires+ 
					//	*(iparm+NPARM*(iCell-1)+31);
					*(iACTvOld[3]+nACTvOld)=ires+ 
					*(iparm+NPARM*(mCell-1)+31)+*(mapSpeed[jz0]+jy0*NI+jx0);
					*(iXCTapd[jz0]+jy0*NI+jx0)+=1;
					} else {
					*(iXCTapd[jz0]+jy0*NI+jx0)=0;
					}
					}
					// <<<<<<<<<<<*/
					if (iSTOP ==1) break;
				}
				if (iSTOP ==1) break;

				if (nStep==0) break; // continue;
				jdist=jdist+1;
				if (jdist>=idist) break; // continue;
				for(k=1;k<=nStep;k++) {
					*(jACTv[0]+k)=*(kACTv[0]+k);
					*(jACTv[1]+k)=*(kACTv[1]+k);
					*(jACTv[2]+k)=*(kACTv[2]+k);
				}
				nbrch=nStep;
				nStep=0;
			}
		}
		//TRACE("\nbreak5 iSTOP=%d ",iSTOP);
		if (iSTOP ==1) break;
		// <------- END of propagation (1000) -------------
		if (excited == 0) break;
		nACTv=mACTv;
		// nblck=nblck+nACTv;

		for(i=1;i<=nACTv;i++) {
			for(j=0;j<4;j++) {
				*(iACTv[j]+i)=*(iACTvOld[j]+i);
			}
		}

	} // END of whole while loop 
	TRACE("\nmxcycle=%d",mxcycle);

	mxcycle++; // hui


	// add HB info
	for (itmp=0; itmp<50*ND; itmp++) {
		for (tmp=0;tmp<NCYCL;tmp++) { 
			vHB[tmp][itmp]=0;
		}
	}

	for (itmp=0; itmp<nHB; itmp++) {
		l=iHB[0][itmp];
		j=iHB[1][itmp];
		k=iHB[2][itmp];
		if (itmp==0) i=*(locXCT[k]+j*NJ+l); // Consider only the point near AV Node
		for (tmp=0;tmp<mxcycle;tmp++) { 
			vHB[tmp][itmp]=*(mapXCTm[tmp]+i);
		}
	}

	// Save 
	CFile f;
	CFileException e;
	//short int index;
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'x');
	//filepath.SetAt(index+2,'c');
	//filepath.SetAt(index+3,'t');
	if (myid==0){
		if (!f.Open( dataPath+"tour.xct ", CFile::modeCreate | CFile::modeWrite, &e )) {
#ifdef _DEBUG
			afxDump << "File could not be opened " << e.m_cause << "\n";
#endif
		}
		//f.Write(&mxcycle,2);	
		f.Write(&miBN,2);	
		f.Write(&ic,2);
		f.Write(&totalCell,4);

		for(j=0;j<mxcycle;j++) {    
			for(i=0;i<totalCell;i++) f.Write(mapXCTm[j]+i,2);
		}    
		f.Close();
	};
	/*	
	FILE * iow;
	iow=fopen("fpMapXCTm.txt","wt");	
	if (iow == NULL) {		
	fprintf(stderr, "Open .txt for write failed! \n");		
	return;		
	}	

	long temploc;

	temploc=*(locXCT[45]+22*NJ+33);
	fprintf(iow,"33 22 45 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[40]+30*NJ+32);
	fprintf(iow,"32 30 40 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[48]+20*NJ+30);
	fprintf(iow,"30 20 48 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[56]+8*NJ+26);
	fprintf(iow,"26  8 56 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[62]+10*NJ+21);
	fprintf(iow,"21 10 62 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[62]+30*NJ+13);
	fprintf(iow,"13 30 62 %3d\n",*(mapXCTm[0]+temploc));

	for(l=0;l<mxcycle;l++) {    
	fprintf(iow,"l=%d\n",l);
	for(k=0;k<NK;k++) {
	for(j=0;j<NJ;j++) {
	for(i=NI-1;i>-1;i--) {
	temploc = *(locXCT[k]+j*NJ+i);
	if (temploc < 0) fprintf(iow,"    ");
	else fprintf(iow,"%3d ",*(mapXCTm[l]+temploc));
	}
	fprintf(iow,"j=%d\n",j);
	}
	fprintf(iow,"k=%d\n",k);
	}
	}
	fclose(iow);
	*/

	for(i=0;i<4;i++) {
		free(iACTv[i]);
		free(iACTvOld[i]);
		free(jACTv[i]);
		free(kACTv[i]);
	}       
	for(i=0;i<NK;i++) {
		free(iXCT[i]);
		free(iXCTapd[i]);
		free(iXCTOld[i]);
	}
}

// ---- BB conduction ----
void bbDLYm(short int i00,short int j00,short int k00) {
	short int ii;
	for(ii=0;ii<nttl;ii++) {
		if (i00!=ipttl[0][ii]) continue;
		if (j00!=ipttl[1][ii]) continue;
		if (k00!=ipttl[2][ii]) continue;
		// Add for BB interval by hui wang
		ibbSTEP=ipttl[3][ii];
		// End of add for BB interval by hui wang
		ibbDLY=100*ND/(ipttl[3][ii]+1);
		if((ibbDLY*ipttl[3][ii])<100*ND) ibbDLY+=1;
	}
}

// ********** find the time since last excitation*********
void rdXCTm(short int icc,short int i00,short int j00,short int k00) {
	short int ncyc,n1cyc;
	idltt=INFTIME;  /*<Comment by ALF> period between 2 continuous excitation*/
	idltc=0;		/*<Comment by ALF> delta of 2 periods */
	short int n;
	long locxct;

	locxct=*(locXCT[k00]+j00*NI+i00);
	if(locxct<0) return;

	for(n=NCYCL-1;n>=0;n--) {
		ncyc=*(mapXCTm[n]+locxct);
		if (icc>=ncyc) { 
			idltt=icc-ncyc; 
			break; 
		} 
	}
	if ((n<=0)||(n>=NCYCL-1)) return;  /*<Comment by ALF> To prevent a over-cross array operation*/
	n1cyc=*(mapXCTm[n+1]+locxct);
	if (n1cyc==INFTIME) return;
	idltc=n1cyc-ncyc-ncyc+*(mapXCTm[n-1]+locxct);
	return;
}

// ******* sub write XCT **********
void wtXCTm(short int icc,short int i00,short int j00,short int k00)
{
	short int n;
	long locxct=*(locXCT[k00]+j00*NI+i00);

	if(locxct<0) return;
	for(n=0;n<NCYCL;n++) {

		if (*(mapXCTm[n]+locxct)!=INFTIME) continue; 
		*(mapXCTm[n]+locxct)=icc; 
		if (mxcycle<n) mxcycle=n;
		break;
	}
	return;
}


// Body surface potential calculation
void BSPcalm(void) {
	//-------------------- modified by ALF at 2008-8-19 begin -------------------->
	//modified
	void BSPitmm(short int, short int **, float *, float *, float *, float *, float **, float **,short int, float *, float *);
	//void BSPitmm(short int, short int **, float *, float *, float *, float *, float *, float *);
	vector<float> epicPOT[TSTEP]; // potential 
	//-------------------- modified by ALF at 2008-8-19 end --------------------< 


	short int nVCG,BSPm,mTime,iTime,i,j;
	short int nsnrt;
	float *VCGs[3],*VCGs_reduce[3];//*VCGs_reduce[3] by sf 090622
	float eff;
	float *endoHnnA;
	float *endoPOT[TSTEP];//*endoPOT_reduce[TSTEP];//*endoPOT_reduce[TSTEP] by sf 090622
	HFILE hFp;      
	short int index;
	int nn,n0,n1,n2,ni;
	float pi=3.14159;
	short int *tnd[3]; 

	//int  myid, numprocs;
	int numprocs = 2;
    int  namelen;

    //fprintf(stdout,"BSPcalm *** Process %d of %d is on %s\n", myid, numprocs, processor_name);
	 //fflush(stdout);

	for(i=0;i<TSTEP;i++) {
		endoPOT[i]=(float *) malloc(2*NENDO*ND3*4);
		//endoPOT_reduce[i]=(float *) malloc(2*NENDO*ND3*4);//by sf 090622
		if(endoPOT[i]==NULL) {
			MessageBox(NULL,"Out of memory !",NULL,MB_OK);
			return;// 0;
		}
	}
	for(i=0;i<TSTEP;i++) {
		for(ni=0;ni<2*NENDO*ND3;ni++) {
			*(endoPOT[i]+ni)=(float)0;
			//*(endoPOT_reduce[i]+ni)=(float)0;
		}
	}
	endoHnnA=(float *) malloc(2*NENDO*ND3*4);
	if((endoHnnA==NULL)) {
		MessageBox(NULL,"Out of memory !",NULL,MB_OK);
		return;// 0;
	}  
	for(ni=0;ni<2*NENDO*ND3;ni++) *(endoHnnA+ni)=(float)0;

	//-------------------- modified by ALF at 2008-8-19 begin -------------------->
	//add: malloc epicardial potential array
	ASSERT(Nepic != 0);
	for (i=0; i<TSTEP; ++i) {
		epicPOT[i].resize(Nepic,0);
	}
	//-------------------- modified by ALF at 2008-8-19 end --------------------< 

	for(i=0;i<3;i++) {
		VCGs[i]=(float *)malloc(TSTEP*4);
		VCGs_reduce[i]=(float *)malloc(TSTEP*4);//by sf 090622
		if (VCGs[i]==NULL) {
			MessageBox(NULL,"Out of memory !",NULL,MB_OK);
			flag_flop=1;
			return;
		}
	}
	for (i=0;i<3;i++) VCG[i]=(float)0;    
	for (i=0;i<3;i++) {
		for (j=0;j<TSTEP;j++) {
			*(VCGs[i]+j)=(float)0;
			*(VCGs_reduce[i]+j)=(float)0;//by sf 090622
		}
	}
	// matrix data of endo-body surface
	for(i=0;i<3;i++) {
		tnd[i] = (short int *) malloc((NL-2)*2*2);
		if(tnd[i]==NULL) {
			MessageBox(NULL,"Out of memory !",NULL,MB_OK);
			return;// 0;
		}
	}    
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'t');
	//filepath.SetAt(index+2,'n');
	//filepath.SetAt(index+3,'d');	
	//     hFp=_lopen("trstrs.5",READ);
	//hFp=_lopen(filepath,OF_READ); 
	hFp=_lopen(dataPath+"tour.tnd ",OF_READ);

	if (hFp==HFILE_ERROR) {
		//MessageBox(NULL,"Can not open tnd model file !",NULL,MB_OK);
		fprintf(stdout,"Can not open tnd file ! !\n");
		fflush(stdout);

		flag_flop=1;
		return;
	}	
	for(i=0;i<(NL-2)*2;i++) {
		_lread(hFp,tnd[0]+i,2);
		_lread(hFp,tnd[1]+i,2);
		_lread(hFp,tnd[2]+i,2);
	}
	_lclose(hFp);    


	float ax,ay,az,x0,y0,z0,x1,y1,z1,x2,y2,z2,a01,a12,a20,s,h;
	float x3,y3,z3,u3,x4,y4,z4,u4,x5,y5,z5,u5;
	short int ISGN=1;
	float *hnn;
	hnn=(float *) malloc((NL-2)*2*(NL-2)*2*4);
	if (hnn==NULL) {
		MessageBox(NULL,"Out of memory !",NULL,MB_OK);
		return;// 0;
	}  
	for(ni=0;ni<(NL-2)*2*(NL-2)*2;ni++) {
		*(hnn+ni)=(float)0;
	}
	ni=0;
	for(nn=0;nn<(NL-2)*2;nn++) {
		// ---- measurement location -------
		n0=*(tnd[0]+nn)-1;  /*<Comment by ALF> triangle node array */
		n1=*(tnd[1]+nn)-1;
		n2=*(tnd[2]+nn)-1;
		ax=(*(r[0]+n0)+*(r[0]+n1)+*(r[0]+n2))/3;  /*<Comment by ALF> distance from center of triangle to view point */
		ay=(*(r[1]+n0)+*(r[1]+n1)+*(r[1]+n2))/3;
		az=(*(r[2]+n0)+*(r[2]+n1)+*(r[2]+n2))/3;
		for(i=0;i<(NL-2)*2;i++) {
			if (i==nn) {
				*(hnn+ni)=0.5;
				ni++;
				continue;
			}
			/*<Comment by ALF> xn,yn,zn is the co-ordinate by set the center of triangle as the the origin*/
			n0=*(tnd[0]+i)-1;   
			x0=*(r[0]+n0)-ax;
			y0=*(r[1]+n0)-ay;
			z0=*(r[2]+n0)-az;
			n1=*(tnd[1]+i)-1;
			x1=*(r[0]+n1)-ax;
			y1=*(r[1]+n1)-ay;
			z1=*(r[2]+n1)-az;
			n2=*(tnd[2]+i)-1;
			x2=*(r[0]+n2)-ax;
			y2=*(r[1]+n2)-ay;
			z2=*(r[2]+n2)-az;
			a01=acos((x0*x1+y0*y1+z0*z1)/sqrt(x0*x0+y0*y0+z0*z0)/sqrt(x1*x1+y1*y1+z1*z1));
			a12=acos((x1*x2+y1*y2+z1*z2)/sqrt(x1*x1+y1*y1+z1*z1)/sqrt(x2*x2+y2*y2+z2*z2));
			a20=acos((x2*x0+y2*y0+z2*z0)/sqrt(x2*x2+y2*y2+z2*z2)/sqrt(x0*x0+y0*y0+z0*z0));
			s=(a01+a12+a20)/2;
			h=tan(s/2)*tan((s-a01)/2)*tan((s-a12)/2)*tan((s-a20)/2);
			if (h<0) h=-h;
			s=sqrt(h);
			h=atan(s)/pi;			
			*(hnn+ni)=-h;
			ni++;
		}
	}

	float *endoHnnB,*endoHnnC;
	endoHnnB=(float *) malloc(NENDO*ND3*(NL-2)*2*4);
	endoHnnC=(float *) malloc(NENDO*ND3*(NL-2)*2*4);
	if ((endoHnnB==NULL)||(endoHnnC==NULL)) {
		MessageBox(NULL,"Out of memory !",NULL,MB_OK);
		return;// 0;
	}  
	for(ni=0;ni<NENDO*ND3*(NL-2)*2;ni++) {
		*(endoHnnB+ni)=(float)0;
		*(endoHnnC+ni)=(float)0;
	}

	for(nn=0;nn<NendoB;nn++) {
		// ---- measurement location -------
		ax=HRTx0+endoBx[nn]*tmswf[0][0]+endoBy[nn]*tmswf[0][1]+endoBz[nn]*tmswf[0][2];
		ay=HRTy0+endoBx[nn]*tmswf[1][0]+endoBy[nn]*tmswf[1][1]+endoBz[nn]*tmswf[1][2];
		az=HRTz0+endoBx[nn]*tmswf[2][0]+endoBy[nn]*tmswf[2][1]+endoBz[nn]*tmswf[2][2];
		for(i=0;i<(NL-2)*2;i++) {
			n0=*(tnd[0]+i)-1;
			x0=*(r[0]+n0)-ax;
			y0=*(r[1]+n0)-ay;
			z0=*(r[2]+n0)-az;
			n1=*(tnd[1]+i)-1;
			x1=*(r[0]+n1)-ax;
			y1=*(r[1]+n1)-ay;
			z1=*(r[2]+n1)-az;
			n2=*(tnd[2]+i)-1;
			x2=*(r[0]+n2)-ax;
			y2=*(r[1]+n2)-ay;
			z2=*(r[2]+n2)-az;
			x3=(z2*x0-z0*x2)*(x0*y1-x1*y0)-(x2*y0-x0*y2)*(z0*x1-z1*x0);
			y3=(x2*y0-x0*y2)*(y0*z1-y1*z0)-(y2*z0-y0*z2)*(x0*y1-x1*y0);
			z3=(y2*z0-y0*z2)*(z0*x1-z1*x0)-(z2*x0-z0*x2)*(y0*z1-y1*z0);
			u3=(y2*z0-y0*z2)*(y0*z1-y1*z0)+(z2*x0-z0*x2)*(z0*x1-z1*x0)+(x2*y0-x0*y2)*(x0*y1-x1*y0);
			a01=atan(-sqrt(x3*x3+y3*y3+z3*z3)/u3);
			x4=(z0*x1-z1*x0)*(x1*y2-x2*y1)-(x0*y1-x1*y0)*(z1*x2-z2*x1);
			y4=(x0*y1-x1*y0)*(y1*z2-y2*z1)-(y0*z1-y1*z0)*(x1*y2-x2*y1);
			z4=(y0*z1-y1*z0)*(z1*x2-z2*x1)-(z0*x1-z1*x0)*(y1*z2-y2*z1);
			u4=(y0*z1-y1*z0)*(y1*z2-y2*z1)+(z0*x1-z1*x0)*(z1*x2-z2*x1)+(x0*y1-x1*y0)*(x1*y2-x2*y1);
			a12=atan(-sqrt(x4*x4+y4*y4+z4*z4)/u4);
			x5=(z1*x2-z2*x1)*(x2*y0-x0*y2)-(x1*y2-x2*y1)*(z2*x0-z0*x2);
			y5=(x1*y2-x2*y1)*(y2*z0-y0*z2)-(y1*z2-y2*z1)*(x2*y0-x0*y2);
			z5=(y1*z2-y2*z1)*(z2*x0-z0*x2)-(z1*x2-z2*x1)*(y2*z0-y0*z2);
			u5=(y1*z2-y2*z1)*(y2*z0-y0*z2)+(z1*x2-z2*x1)*(z2*x0-z0*x2)+(x1*y2-x2*y1)*(x2*y0-x0*y2);
			a20=atan(-sqrt(x5*x5+y5*y5+z5*z5)/u5);
			s=(a01+a12+a20-pi)*ISGN; // ISGN=1 only; since ISGN=-1 is impossible in our case
			h=tan(s/2)*tan((s-a01)/2)*tan((s-a12)/2)*tan((s-a20)/2);
			if (h<0) h=-h;
			s=sqrt(h);
			h=atan(s)/pi;			
			*(endoHnnB+nn)=h;
		}
	}
	for(nn=0;nn<NendoC;nn++) {
		// ---- measurement location -------
		ax=HRTx0+endoCx[nn]*tmswf[0][0]+endoCy[nn]*tmswf[0][1]+endoCz[nn]*tmswf[0][2];
		ay=HRTy0+endoCx[nn]*tmswf[1][0]+endoCy[nn]*tmswf[1][1]+endoCz[nn]*tmswf[1][2];
		az=HRTz0+endoCx[nn]*tmswf[2][0]+endoCy[nn]*tmswf[2][1]+endoCz[nn]*tmswf[2][2];
		for(i=0;i<(NL-2)*2;i++) {
			n0=*(tnd[0]+i)-1;
			x0=*(r[0]+n0)-ax;
			y0=*(r[1]+n0)-ay;
			z0=*(r[2]+n0)-az;
			n1=*(tnd[1]+i)-1;
			x1=*(r[0]+n1)-ax;
			y1=*(r[1]+n1)-ay;
			z1=*(r[2]+n1)-az;
			n2=*(tnd[2]+i)-1;
			x2=*(r[0]+n2)-ax;
			y2=*(r[1]+n2)-ay;
			z2=*(r[2]+n2)-az;
			x3=(z2*x0-z0*x2)*(x0*y1-x1*y0)-(x2*y0-x0*y2)*(z0*x1-z1*x0);
			y3=(x2*y0-x0*y2)*(y0*z1-y1*z0)-(y2*z0-y0*z2)*(x0*y1-x1*y0);
			z3=(y2*z0-y0*z2)*(z0*x1-z1*x0)-(z2*x0-z0*x2)*(y0*z1-y1*z0);
			u3=(y2*z0-y0*z2)*(y0*z1-y1*z0)+(z2*x0-z0*x2)*(z0*x1-z1*x0)+(x2*y0-x0*y2)*(x0*y1-x1*y0);
			a01=atan(-sqrt(x3*x3+y3*y3+z3*z3)/u3);
			x4=(z0*x1-z1*x0)*(x1*y2-x2*y1)-(x0*y1-x1*y0)*(z1*x2-z2*x1);
			y4=(x0*y1-x1*y0)*(y1*z2-y2*z1)-(y0*z1-y1*z0)*(x1*y2-x2*y1);
			z4=(y0*z1-y1*z0)*(z1*x2-z2*x1)-(z0*x1-z1*x0)*(y1*z2-y2*z1);
			u4=(y0*z1-y1*z0)*(y1*z2-y2*z1)+(z0*x1-z1*x0)*(z1*x2-z2*x1)+(x0*y1-x1*y0)*(x1*y2-x2*y1);
			a12=atan(-sqrt(x4*x4+y4*y4+z4*z4)/u4);
			x5=(z1*x2-z2*x1)*(x2*y0-x0*y2)-(x1*y2-x2*y1)*(z2*x0-z0*x2);
			y5=(x1*y2-x2*y1)*(y2*z0-y0*z2)-(y1*z2-y2*z1)*(x2*y0-x0*y2);
			z5=(y1*z2-y2*z1)*(z2*x0-z0*x2)-(z1*x2-z2*x1)*(y2*z0-y0*z2);
			u5=(y1*z2-y2*z1)*(y2*z0-y0*z2)+(z1*x2-z2*x1)*(z2*x0-z0*x2)+(x1*y2-x2*y1)*(x2*y0-x0*y2);
			a20=atan(-sqrt(x5*x5+y5*y5+z5*z5)/u5);
			s=(a01+a12+a20-pi)*ISGN; // ISGN=1 only; since ISGN=-1 is impossible in our case
			h=tan(s/2)*tan((s-a01)/2)*tan((s-a12)/2)*tan((s-a20)/2);
			if (h<0) h=-h;
			s=sqrt(h);
			h=atan(s)/pi;			
			*(endoHnnC+nn)=h;
		}
	}
	TRACE("\nNendoB=%d NendoC=%d",NendoB,NendoC);

	//-------------------- modified by ALF at 2008-8-19 begin -------------------->
	//add: store the solid angle of epicardial triangle 
	vector<float> epicHnn;
	epicHnn.resize(Nepic*ND3*(NL-2)*2, 0);
	for(nn=0;nn<Nepic;nn++) {
		// ---- measurement location -------
		ax=HRTx0+epicX[nn]*tmswf[0][0]+epicY[nn]*tmswf[0][1]+epicZ[nn]*tmswf[0][2];
		ay=HRTy0+epicX[nn]*tmswf[1][0]+epicY[nn]*tmswf[1][1]+epicZ[nn]*tmswf[1][2];
		az=HRTz0+epicX[nn]*tmswf[2][0]+epicY[nn]*tmswf[2][1]+epicZ[nn]*tmswf[2][2];
		for(i=0;i<(NL-2)*2;i++) {
			n0=*(tnd[0]+i)-1;
			x0=*(r[0]+n0)-ax;
			y0=*(r[1]+n0)-ay;
			z0=*(r[2]+n0)-az;
			n1=*(tnd[1]+i)-1;
			x1=*(r[0]+n1)-ax;
			y1=*(r[1]+n1)-ay;
			z1=*(r[2]+n1)-az;
			n2=*(tnd[2]+i)-1;
			x2=*(r[0]+n2)-ax;
			y2=*(r[1]+n2)-ay;
			z2=*(r[2]+n2)-az;
			x3=(z2*x0-z0*x2)*(x0*y1-x1*y0)-(x2*y0-x0*y2)*(z0*x1-z1*x0);
			y3=(x2*y0-x0*y2)*(y0*z1-y1*z0)-(y2*z0-y0*z2)*(x0*y1-x1*y0);
			z3=(y2*z0-y0*z2)*(z0*x1-z1*x0)-(z2*x0-z0*x2)*(y0*z1-y1*z0);
			u3=(y2*z0-y0*z2)*(y0*z1-y1*z0)+(z2*x0-z0*x2)*(z0*x1-z1*x0)+(x2*y0-x0*y2)*(x0*y1-x1*y0);
			a01=atan(-sqrt(x3*x3+y3*y3+z3*z3)/u3);
			x4=(z0*x1-z1*x0)*(x1*y2-x2*y1)-(x0*y1-x1*y0)*(z1*x2-z2*x1);
			y4=(x0*y1-x1*y0)*(y1*z2-y2*z1)-(y0*z1-y1*z0)*(x1*y2-x2*y1);
			z4=(y0*z1-y1*z0)*(z1*x2-z2*x1)-(z0*x1-z1*x0)*(y1*z2-y2*z1);
			u4=(y0*z1-y1*z0)*(y1*z2-y2*z1)+(z0*x1-z1*x0)*(z1*x2-z2*x1)+(x0*y1-x1*y0)*(x1*y2-x2*y1);
			a12=atan(-sqrt(x4*x4+y4*y4+z4*z4)/u4);
			x5=(z1*x2-z2*x1)*(x2*y0-x0*y2)-(x1*y2-x2*y1)*(z2*x0-z0*x2);
			y5=(x1*y2-x2*y1)*(y2*z0-y0*z2)-(y1*z2-y2*z1)*(x2*y0-x0*y2);
			z5=(y1*z2-y2*z1)*(z2*x0-z0*x2)-(z1*x2-z2*x1)*(y2*z0-y0*z2);
			u5=(y1*z2-y2*z1)*(y2*z0-y0*z2)+(z1*x2-z2*x1)*(z2*x0-z0*x2)+(x1*y2-x2*y1)*(x2*y0-x0*y2);
			a20=atan(-sqrt(x5*x5+y5*y5+z5*z5)/u5);
			s=(a01+a12+a20-pi)*ISGN; // ISGN=1 only; since ISGN=-1 is impossible in our case
			h=tan(s/2)*tan((s-a01)/2)*tan((s-a12)/2)*tan((s-a20)/2);
			if (h<0) h=-h;
			s=sqrt(h);
			h=atan(s)/pi;
			epicHnn[nn] = h;
		}
	}
	//-------------------- modified by ALF at 2008-8-19 end --------------------< 


	/*	for(nn=0;nn<NendoB;nn++) {
	// ---- measurement location -------
	ax=HRTx0+endoBx[nn]*tmswf[0][0]+endoBy[nn]*tmswf[0][1]+endoBz[nn]*tmswf[0][2];
	ay=HRTy0+endoBx[nn]*tmswf[1][0]+endoBy[nn]*tmswf[1][1]+endoBz[nn]*tmswf[1][2];
	az=HRTz0+endoBx[nn]*tmswf[2][0]+endoBy[nn]*tmswf[2][1]+endoBz[nn]*tmswf[2][2];
	for(i=0;i<(NL-2)*2;i++) {
	n0=*(tnd[0]+i)-1;
	x0=*(r[0]+n0)-ax;
	y0=*(r[1]+n0)-ay;
	z0=*(r[2]+n0)-az;
	n1=*(tnd[1]+i)-1;
	x1=*(r[0]+n1)-ax;
	y1=*(r[1]+n1)-ay;
	z1=*(r[2]+n1)-az;
	n2=*(tnd[2]+i)-1;
	x2=*(r[0]+n2)-ax;
	y2=*(r[1]+n2)-ay;
	z2=*(r[2]+n2)-az;
	a01=acos((x0*x1+y0*y1+z0*z1)/sqrt(x0*x0+y0*y0+z0*z0)/sqrt(x1*x1+y1*y1+z1*z1));
	a12=acos((x1*x2+y1*y2+z1*z2)/sqrt(x1*x1+y1*y1+z1*z1)/sqrt(x2*x2+y2*y2+z2*z2));
	a20=acos((x2*x0+y2*y0+z2*z0)/sqrt(x2*x2+y2*y2+z2*z2)/sqrt(x0*x0+y0*y0+z0*z0));
	s=(a01+a12+a20)/2;
	h=tan(s/2)*tan((s-a01)/2)*tan((s-a12)/2)*tan((s-a20)/2);
	if (h<0) h=-h;
	s=sqrt(h);
	h=atan(s)/pi;			
	*(endoHnnB+nn)=h;
	}
	}
	for(nn=0;nn<NendoC;nn++) {
	// ---- measurement location -------
	ax=HRTx0+endoCx[nn]*tmswf[0][0]+endoCy[nn]*tmswf[0][1]+endoCz[nn]*tmswf[0][2];
	ay=HRTy0+endoCx[nn]*tmswf[1][0]+endoCy[nn]*tmswf[1][1]+endoCz[nn]*tmswf[1][2];
	az=HRTz0+endoCx[nn]*tmswf[2][0]+endoCy[nn]*tmswf[2][1]+endoCz[nn]*tmswf[2][2];
	for(i=0;i<(NL-2)*2;i++) {
	n0=*(tnd[0]+i)-1;
	x0=*(r[0]+n0)-ax;
	y0=*(r[1]+n0)-ay;
	z0=*(r[2]+n0)-az;
	n1=*(tnd[1]+i)-1;
	x1=*(r[0]+n1)-ax;
	y1=*(r[1]+n1)-ay;
	z1=*(r[2]+n1)-az;
	n2=*(tnd[2]+i)-1;
	x2=*(r[0]+n2)-ax;
	y2=*(r[1]+n2)-ay;
	z2=*(r[2]+n2)-az;
	a01=acos((x0*x1+y0*y1+z0*z1)/sqrt(x0*x0+y0*y0+z0*z0)/sqrt(x1*x1+y1*y1+z1*z1));
	a12=acos((x1*x2+y1*y2+z1*z2)/sqrt(x1*x1+y1*y1+z1*z1)/sqrt(x2*x2+y2*y2+z2*z2));
	a20=acos((x2*x0+y2*y0+z2*z0)/sqrt(x2*x2+y2*y2+z2*z2)/sqrt(x0*x0+y0*y0+z0*z0));
	s=(a01+a12+a20)/2;
	h=tan(s/2)*tan((s-a01)/2)*tan((s-a12)/2)*tan((s-a20)/2);
	if (h<0) h=-h;
	s=sqrt(h);
	h=atan(s)/pi;			
	*(endoHnnC+nn)=h;
	}
	}
	TRACE("\nNendoB=%d NendoC=%d",NendoB,NendoC);
	*/

	//---- body surface potential distribution [time(msec)]
	nsnrt=mBCL;
	//    nsnrt=nsnrt*3; August 10, 1996
	nTimeStep=0;
	nVCG=0;//nVCG_old=0;//by sf 090401
	itbuf=0;
	bufGRD=(float)0;
	for(short int n=0;n<2;n++) {
		for(short int m=0;m<3;m++) {
			bufVCG[n][m]=(float)0;
		}
	}
	mTime=maxXctStep*3;
	//iTime=3*ND;
	//// by sf 090403-6
	//	while (iTime <= mTime) {		
	//	BSPitmm_gatherijk(iTime, tnd, hnn, endoHnnA, endoHnnB, endoHnnC,endoPOT,VCGs,nsnrt);
	//	iTime=iTime+3;//*tnum;iTime_old=iTime_old+3;//iTime=iTime+3;//iTime=iTime+nextStep;		
	//};
	iTime=3*ND;//printf("mTime1=%d",(mTime/3+2));
	gatheralldpl   =   (float**)malloc((mTime/3+1)*sizeof(float)); //for 1--mTime/3
	gatherallijk   =   (int**)malloc((mTime/3+1)*sizeof(int));
	countallijk   =   (int*)malloc((mTime/3+1)*sizeof(int));countallijk_reduce   =   (int*)malloc((mTime/3+1)*sizeof(int));
	for(i=0;i<=mTime;i=i+3) 
	{
		*(countallijk+i/3)=0;  //by sf 090621
		*(countallijk_reduce+i/3)=0;
	}
	// isumloops记录每个iTime循环次数的总和
	//	iloops[3]0行: [0,0]存dipole总数,其余存dipole数量
	//        1行: 对dipole数量排序
	//        2行:排序后dipole对应的iTime序号
	//itask[2]0行:存各MPI进程的任务分配的进程号
	//        1行:每个进程分别各自存分配到的iTime号,其中0元素存分到的任务个数

	for(i=0;i<2;i=i+1)
	{
		itask[i]   =   (int *)malloc((mTime/3+1)*sizeof(int));
		for(j=0;j<=mTime;j=j+3)
		{
			*(itask[i]+j/3)=777;
		}
	};
	for(i=0;i<3;i=i+1) 
	{
		iloops[i]   =   (int *)malloc((mTime/3+1)*sizeof(int));
		for(j=0;j<=mTime;j=j+3)
		{
			*(iloops[i]+j/3)=-8;
		}

	}
	for(j=0;j<=mTime;j=j+3)
	{
		*(iloops[2]+j/3)=j;
	}


	//printf("before BSPitmm-begin=%f,\n",(bsptime[0]-starttime)/CLK_TCK);
	iTimebegin=1,iTimeend=mTime/3;
	int loopnum=0,loop=0;

	//for(loop=1;loop<=iTimeend;loop++)
	//{
	//	if (myid==0) printf("%d,%d,%d,iloops test loop=%d,myid=%d\n",*(iloops[0]+loop)*3,*(iloops[1]+loop)*3,*(iloops[2]+loop),loop*3,myid);
	//}

	bsptime[0] =clock();

#pragma omp parallel for // OpenMP--begin //by sf 090621计算每个iTime的dipole数量
	for(i=iTime;i<=mTime;i=i+3)
	{
		int tid=omp_get_thread_num(),tnum=omp_get_num_threads();
		corenum=tnum;
		*(iloops[0]+i/3)=BSPitmmcount(i);
		*(iloops[1]+i/3)=*(iloops[0]+i/3);//为了后面dipole排序
		//isumdipoles=isumdipoles+*(iloops[0]+i/3);//myid=0计算的dipole和其他节点不一样,why?god save me;查名原因:多线程共享,累加出错
		*(itask[0]+i/3)=i;
		if (*(iloops[0]+i/3)>0)
		{
			//printf("malloc j=%d,myid=%d,*(countallijk+i/3)=%d,*(countallijk_reduce+j/3)=%d\n",j,myid,*(countallijk+j/3),*(countallijk_reduce+j/3));
			gatherallijk[i/3]   =   (int   *)   malloc(   *(iloops[0]+i/3)*3*sizeof(int)   ); 
			gatheralldpl[i/3]   =   (float   *)   malloc(   *(iloops[0]+i/3)*3*sizeof(float)   ); 
		    for(j=0;j<*(iloops[0]+i/3);j=j+1)
			{
				*(gatherallijk[i/3]+j)=0;
				*(gatheralldpl[i/3]+j)=float(0.0);
			};
		}

		//printf("%d iloops\n",*(iloops[0]+i/3));//printf("iTime=%d,loopcount=%d\n",loop,*(iloops+loop/3));
	}
// OpenMP--end
//dipole排序      *(iloops[1]+0)存放暂存数据
	for(i=1;i<=iTimeend;i++) 
	{
		//printf("i=%d,iloops[0]=%d,itask[0]=%d,itask[1]=%d,myid=%d\n",i,*(iloops[0]+i),*(itask[0]+i),*(itask[1]+i),myid);
		isumdipoles=isumdipoles+*(iloops[0]+i);
	}
	for(loop=1;loop<=iTimeend;loop++)
	{
	 loopnum=loop;
	 for(i=loop+1;i<=iTimeend;i++)
		 if (*(iloops[1]+loopnum)<*(iloops[1]+i)) loopnum=i;
	 *(iloops[1]+0)=*(iloops[1]+loopnum);*(iloops[2]+0)=*(iloops[2]+loopnum);
	 *(iloops[1]+loopnum)=*(iloops[1]+loop);*(iloops[2]+loopnum)=*(iloops[2]+loop);
	 *(iloops[1]+loop)=*(iloops[1]+0);*(iloops[2]+loop)=*(iloops[2]+0);
	};

//*///091212--100211 sumdipole=dipole总数,dipolep=平均dipole数,dipole0=MPI进程0 dipole数,
	//iteration0=MPI进程0 iteration数,turn=0,mTimeby0=[1,mTimeby0]--其他进程;(mTimeby0,iTimeEND]--进程0,count=1,head=计数
	int sumdipole=0,dipolep=0,dipole0=0,iteration0=0,turn=0,mTimeby0=0,count=1,head=1;
	int tail,tailbegin,tailend,exi,ldipole,count1; //[1,gpuend) [gpuend,cpuend](cpuend,tail)[tail,*(itask[1]+0)]
	for(loop=1;loop<=iTimeend;loop++)//算dipole总数
	{
		sumdipole=sumdipole+*(iloops[0]+loop);

	};
	dipolep=sumdipole/numprocs;
	dipole0=0;
	for(loop=iTimeend;loop>=1;loop--)//给进程0分配尾部数量<=平均dipole的iteration
	{
		dipole0=dipole0+*(iloops[1]+loop);
        if (dipole0 > dipolep) 
		{
			break;
		};
			*(itask[0]+loop)=0;
	};
	mTimeby0=loop;//未分配的任务包含mTimeby0  [1,mTimeby0]给其他进程

	turn=1;
	for(loop=1;loop<=mTimeby0;loop++)
	{
		*(itask[0]+loop)=turn;
		turn++;
		if (turn>numprocs-1) turn=1;
	};
	int msumdipole=0,mdipolep=0,tdipole=0,predictend=0,gpuend=1,cpuend=iTimeend;
	//msumdipole=每个MPI进程分配到的dipole数,mdipolep=每个线程的平均dipole数,
	//gpuspeed=gpu加速比,tdipole临时值,predictend=预测动态调度中间点(t0和其他线程相遇点);
	//[1,gpuend),(cpuend,iTimeend]静态调度;[gpuend,cpuend]动态调度,数量:cpuend-gpuend+1;
	head=1;
	for(loop=1;loop<=iTimeend;loop++)
	{	
		//printf("i=%d,iloops[0]=%d,itask[0]=%d,itask[1]=%d,myid=%d\n",loop,*(iloops[0]+loop),*(itask[0]+loop),*(itask[1]+loop),myid);
		//if (myid==*(itask[0]+loop))
		//{
			*(itask[1]+head)=*(iloops[2]+loop);
			head++;
			msumdipole=msumdipole+*(iloops[0]+*(itask[1]+loop)/3);

		//};
	};
	*(itask[1]+0)=head-1;//*(itask[1])每个进程保存自己的iTime,*(itask[1]+0)保存最后一个任务的序号,也是任务个数
	// mdipolep O到P分到的diploe数量
	mdipolep=msumdipole*gpuspeed/(gpuspeed+corenum-1);
	for(loop=1;loop<=*(itask[1]+0);loop++)
	{
		// tdipole线程线程数量
		tdipole=tdipole+*(iloops[0]+*(itask[1]+loop)/3);
		if (tdipole>=mdipolep) break;
	};
	// predictend就是论文中的P
	predictend=loop-1;
//	printf("gpuspeed=%d,corenum=%d,loop=%d,predictend=%d,myid=%d,tdipole=%d,mdipolep=%d\n",gpuspeed,corenum,loop,predictend,myid,tdipole,mdipolep);
 
	//for(i=1;i<=iTimeend;i++) 
	//{
	//	printf("i=%d,iloops[0]=%d,itask[0]=%d,itask[1]=%d,myid=%d\n",i,*(iloops[0]+i),*(itask[0]+i),*(itask[1]+i),myid);
	//}

	int gwindow=1,cwindow=3;
	tail=*(itask[1]+0);
	gpuend=predictend-gwindow;//3和6可以自己定
	cpuend=predictend+(corenum-1)*cwindow;

	tdipole=0;
	// P点diploe数
	ldipole=*(iloops[0]+*(itask[1]+predictend)/3)*1.2;
	for(loop=*(itask[1]+0);loop>=1;loop--)
	{
		//printf("e loop=%d,tail=%d,*(itask[1]+loop)=%d,*(itask[1]+tail)=%d,myid=%d,tdipole=%d,ldipole=%d\n",loop,tail,*(itask[1]+loop),*(itask[1]+tail),myid,tdipole,ldipole);
		// 
		if ( *(iloops[0]+*(itask[1]+loop)/3) > 30 )
		{
			tdipole=tdipole+*(iloops[0]+*(itask[1]+loop)/3);
			if (tdipole > ldipole)  break;

			exi=*(itask[1]+loop);
			*(itask[1]+loop)=*(itask[1]+tail);
			*(itask[1]+tail)=exi;
			//printf("exi loop=%d,tail=%d,*(itask[1]+loop)=%d,*(itask[1]+tail)=%d,myid=%d,tdipole=%d,ldipole=%d\n",loop,tail,*(itask[1]+loop),*(itask[1]+tail),myid,tdipole,ldipole);
			tail--;

		}

	};

	if (gpuend<=0) gpuend=1;
	if (cpuend>*(itask[1]+0)) cpuend=*(itask[1]+0);
	if (tail <= cpuend) tail=cpuend+1;
	//printf("dipolep=%d,sumdipole=%d,MPI-its=%d,msumdipole=%d,myid=%d,pre-end=%d,gpuend=%d,cpuend=%d\n",dipolep,sumdipole,*(itask[1]+0),msumdipole,myid,predictend,gpuend,cpuend);
	
	if (threadnum>0) omp_set_num_threads(threadnum);
	count=gpuend;iTimebegin=gpuend;iTimeend=cpuend;
	count1=tail;tailbegin=tail;tailend=*(itask[1]+0);
	//printf("tail=%d,tailbegin=%d,tailend=%d,ldipole=%d,count1=%d,myid=%d\n",tail,tailbegin,tailend,ldipole,count1,myid);

	double MPItimebegin=clock(),MPItimeend=0.0;

#pragma omp parallel  //private(iTime) //预跑+动态 //by sf 090828 OpenMP--begin
{  
	int tid=omp_get_thread_num(),tnum=omp_get_num_threads(),myloop,myiTime;

	int s=2*tid-1,k=2*(tnum-1);
	double tbegin=clock(),tend=0.0;
    int dipolesum=0,iterationsum=0;

	if (useGPU==1 && tid==0 && GPUnum>0) 
	{
		gpu_transdata(epicX_old,epicY_old,epicZ_old,tnd,r,rn,endoBx,endoBy,endoBz,endoCx,endoCy,endoCz,tmswf);
	}
	threadnum=tnum;

	if (tid==0)
	{
		myloop=1;//myloop是计数器,是位置,坐标,不乘3
		while (myloop < gpuend)
		{
			// 获得偶侄子数量
			dipolesum=dipolesum+*(iloops[0]+*(itask[1]+myloop)/3);iterationsum++;
			BSPitmm(*(itask[1]+myloop), tnd, hnn, endoHnnA, endoHnnB, endoHnnC,endoPOT,VCGs,nsnrt, &epicHnn[0], &epicPOT[*(itask[1]+myloop)/3-1][0]);
			myloop++;
		};
	}
	else
	{
		myloop=(tail-1)-(tid-1);
		while (myloop > cpuend)
		{
			dipolesum=dipolesum+*(iloops[0]+*(itask[1]+myloop)/3);iterationsum++;
			BSPitmm(*(itask[1]+myloop), tnd, hnn, endoHnnA, endoHnnB, endoHnnC,endoPOT,VCGs,nsnrt, &epicHnn[0], &epicPOT[*(itask[1]+myloop)/3-1][0]);
			s=k-s;myloop=myloop-s;
		};

	};
	//fprintf(stdout,"static over,tid=%d,myid=%d,dipolesum=%d,iterationsum=%d\n", tid,myid,dipolesum,iterationsum);

	#pragma omp critical
	{   
		myiTime=count;count++;
		if (useGPU==1 && tid==0)
			{
				myloop=iTimebegin;iTimebegin++;
			}
		else
			{
				myloop=iTimeend;iTimeend--;
			};
	}
	while (myiTime <= cpuend)
	{
		dipolesum=dipolesum+*(iloops[0]+*(itask[1]+myloop)/3);iterationsum++;
		BSPitmm(*(itask[1]+myloop), tnd, hnn, endoHnnA, endoHnnB, endoHnnC,endoPOT,VCGs,nsnrt, &epicHnn[0], &epicPOT[*(itask[1]+myloop)/3-1][0]);
		#pragma omp critical
		{   
			myiTime=count;count++;
			if (useGPU==1 && tid==0 )
				{
					myloop=iTimebegin;iTimebegin++;
				}
			else
				{
					myloop=iTimeend;iTimeend--;
				};
		}
	};

	#pragma omp critical
	{   
		myiTime=count1;count1++;
		if (useGPU==1 && tid==0)// && GPUnum>0)
			{
				myloop=tailbegin;tailbegin++;
			}
		else
			{
				myloop=tailend;tailend--;
			};

	}
	while (myiTime <= *(itask[1]+0))
	{
		dipolesum=dipolesum+*(iloops[0]+*(itask[1]+myloop)/3);iterationsum++;
		BSPitmm(*(itask[1]+myloop), tnd, hnn, endoHnnA, endoHnnB, endoHnnC,endoPOT,VCGs,nsnrt, &epicHnn[0], &epicPOT[*(itask[1]+myloop)/3-1][0]);
		#pragma omp critical
		{   
			myiTime=count1;count1++;
			if (useGPU==1 && tid==0)// && GPUnum>0)
				{
					myloop=tailbegin;tailbegin++;
				}
			else
				{
					myloop=tailend;tailend--;
				};

		}
	};

	//
	tend=clock();
	//fprintf(stdout,"!threadtime = %f,tid=%d,myid=%d,dipolesum=%d,iterationsum=%d\n", (tend-tbegin)/CLK_TCK,tid,myid,dipolesum,iterationsum);
#pragma omp barrier
	};//by sf 090828 OpenMP--end

	//fprintf(stdout,"!!!MPItime = %f,myid=%d,*(itask[1]+0)=%d\n", (MPItimeend-MPItimebegin)/CLK_TCK,myid,*(itask[1]+0));

	bsptime[1] =clock();
	nTimeStep=mTime/3;nVCG=nTimeStep;//printf("*iTime=%d,nVCG=%d,nTimeStep=%d,iTime_old=%d\n",iTime,nVCG,nTimeStep,iTime_old);//by sf 090402-6

	for(int iTime=3*ND;iTime<=mTime;iTime=iTime+3) *(iStep+iTime/3)=iTime/ND;  //by sf 090621

	if (numprocs>1)   //为了让1个进程也可以执行
	{
		for(i=0;i<=mTime;i=i+3) 
		{
			*(countallijk+i/3)=*(iloops[0]+i/3)*3;
		};

		float VCGssend[3],POTsend[NL];
	};

	bsptime[2] =clock();
//	if (myid==0){
			CFile f2;
			CFileException e2;
			if (!f2.Open( dataPath+"tour.dpn ", CFile::modeCreate | CFile::modeWrite, &e2 )) {
#ifdef _DEBUG
					afxDump << "File could not be opened " << e2.m_cause << "\n";
#endif
		}
		int idpl;//printf("mTime=%d\n",mTime);
		for(iTime=3;iTime<=mTime;iTime=iTime+3)
		{
			idpl=*(countallijk+(iTime/3))/3;
			f2.Write(&iTime,2);//f2.Write(&iTime0,2);
			f2.Write(&idpl,2);//f2.Write(&idpl,2);
		};
		f2.Close();

		CFile f3;
		CFileException e3;
		if (!f3.Open(dataPath+"tour.dpl ", CFile::modeCreate | CFile::modeWrite, &e3 )) {
#ifdef _DEBUG
				afxDump << "File could not be opened " << e3.m_cause << "\n";
#endif
		}

	short int ii,jj,kk;
	for(iTime=3;iTime<=mTime;iTime=iTime+3)
	{
		f3.Write(&iTime,2);//f.Write(&iTime0,2);//from line 4070
		for(j=0;j<*(countallijk+iTime/3);j=j+3)
		{			
			ii=*(gatherallijk[iTime/3]+j);f3.Write(&ii,2);//f.Write(gatherallijk[iTime/3]+j,sizeof(int));
			jj=*(gatherallijk[iTime/3]+j+1);f3.Write(&jj,2);//f.Write(gatherallijk[iTime/3]+j+1,sizeof(int));
			kk=*(gatherallijk[iTime/3]+j+2);f3.Write(&kk,2);//f.Write(gatherallijk[iTime/3]+j+2,sizeof(int));
			f3.Write(gatheralldpl[iTime/3]+j,4*3);//f.Write(gatherdpl+j,4*3);
		};
	};
	
	f3.Close();

		CFile f4;
		CFileException e4;

		if( !f4.Open( dataPath+"tour.ecp ", CFile::modeCreate | CFile::modeWrite, &e4 ) )
		{
	#ifdef _DEBUG
			afxDump << "File could not be opened " << e4.m_cause << "\n";
	#endif
		}

		f4.Write(&nTimeStep,2);
		for(i=1;i<=nTimeStep;i++) {
			f4.Write(iStep+i,2);	
		}
		f4.Write(&NendoB, 2);
		f4.Write(&NendoC, 2);
		for(i=0;i<NendoB;i++) {
			f4.Write(&endoBx[i], 2);
			f4.Write(&endoBy[i], 2);
			f4.Write(&endoBz[i], 2);
		}
		for(i=0;i<NendoC;i++) {
			f4.Write(&endoCx[i], 2);
			f4.Write(&endoCy[i], 2);
			f4.Write(&endoCz[i], 2);
		}

		for(i=0;i<nTimeStep;i++) {
			for(j=0;j<(NendoB+NendoC);j++) {
				f4.Write(endoPOT[i]+j,4);
			}
		}	
		f4.Close();

		CFile f5;
		CFileException e5;
		if( !f5.Open( dataPath+"tour.vcg ", CFile::modeCreate | CFile::modeWrite, &e5 ) )
		{
	#ifdef _DEBUG
			afxDump << "File could not be opened " << e5.m_cause << "\n";
	#endif
		}
		f5.Write(&nVCG,2);
		for(j=1;j<=nVCG;j++) {
			f5.Write(iStep+j,2);
			for(i=0;i<3;i++)
				f5.Write(VCGs[i]+j,4);
		}
		f5.Close();
	
	// ----- save potential data ------
	
	eff=(float)26.5730; 

		CFile f6;
		CFileException e6;

		if( !f6.Open( dataPath+"tour.bsp ", CFile::modeCreate | CFile::modeWrite, &e6 ) )
		{
	#ifdef _DEBUG
			afxDump << "File could not be opened " << e6.m_cause << "\n";
	#endif
		}      
		f6.Write(&nTimeStep,2);		

		for(i=1;i<=nTimeStep;i++) {
			f6.Write(iStep+i,2);						
		}
		for(i=1;i<=nTimeStep;i++) {
			int n = 0;
			for(n=0;n<NL;n++) {
				BSPm=(short int)(eff*(*(POT[n]+i)));
				f6.Write(&BSPm,2);
			}
		}
		f6.Close();

		CFile f7;
		CFileException e7;
		if( !f7.Open( dataPath+"tour.epc ", CFile::modeCreate | CFile::modeWrite, &e7 ) )
		{
	#ifdef _DEBUG
			afxDump << "File could not be opened " << e7.m_cause << "\n";
	#endif
		}

		f7.Write(&nTimeStep,sizeof(nTimeStep));
		for(i=1;i<=nTimeStep;i++) 
			f7.Write(iStep+i,sizeof(short int));
	
		f7.Write(&Nepic, sizeof(Nepic));
		for(i=0;i<Nepic;i++) {
			f7.Write(&epicX[i], sizeof(short int));
			f7.Write(&epicY[i], sizeof(short int));
			f7.Write(&epicZ[i], sizeof(short int));
		}

		for(i=0;i<nTimeStep;i++) {
			for(j=0;j<Nepic;j++) {
				f7.Write(&epicPOT[i][j],sizeof(float));
			}
		}	
		f7.Close();
//};

	bsptime[3] =clock();

	FILE *fptime;
	fptime=fopen(dataPath+"gputime.txt","a")  ;
//	fprintf(fptime,"!!!MPItime = %f,myid=%d,*(itask[1]+0)=%d\n", (MPItimeend-MPItimebegin)/CLK_TCK,myid,*(itask[1]+0));
	fprintf(stdout,"BSPmitttime=%f,communicate=%f,writefile=%f\n",(bsptime[1]-bsptime[0])/CLK_TCK,(bsptime[2]-bsptime[1])/CLK_TCK,(bsptime[3]-bsptime[2])/CLK_TCK);
	fprintf(fptime,"BSPmitttime=%f,communicate=%f,writefile=%f\n",(bsptime[1]-bsptime[0])/CLK_TCK,(bsptime[2]-bsptime[1])/CLK_TCK,(bsptime[3]-bsptime[2])/CLK_TCK);
	fclose(fptime);

	for(i=0;i<3;i++) {
		free(VCGs[i]);
		free(VCGs_reduce[i]);//by sf 090622
		free(tnd[i]);
	}
	for(i=0;i<TSTEP;i++) {
		free(endoPOT[i]);
	}
	free(hnn);
	free(endoHnnA);
	free(endoHnnB);	
	free(endoHnnC);	
}

//-------------------- modified by ALF at 2008-8-19 begin -------------------->
//modified
void BSPitmm(short int iTime0, short int **tnd,float *hnn, float *endoHnnA, float *endoHnnB, float *endoHnnC,float **endoPOT,float **VCGs,short int nsnrt, float *epicHnn, float *epicPOT) {
	//void BSPitmm(short int iTime0, short int **tnd,float *hnn, float *endoHnnA, float *endoHnnB, float *endoHnnC, float *epicHnn, float *epicPOT) {
	ASSERT(epicHnn != NULL);
	//-------------------- modified by ALF at 2008-8-19 end --------------------< 
	float aptcalm(short int,short int,short int,short int,short int);
	void anfct(short int i, short int j, short int k, float v[3]);

	char iCell;
	const short int OK_SAV=1; 
	short int iseqx[12]={ -1, 0, 0, 1, 1, 0, 1, 0, 0,-1,-1, 0 };
	short int iseqy[12]={  0,-1, 0,-1, 0, 1, 0, 1, 0, 1, 0,-1 };
	short int iseqz[12]={  0, 0,-1, 0,-1,-1, 0, 0, 1, 0, 1, 1 };
	short int nskip=2;
	short int i,j,k,l,ix,iy,iz,icell,l6,jx,jy,jz,jcell;

	int nsum,n;
	int intvl;
	int idpl;

	float asd,add,rtmax,gsum,compm,compp,compo,ax,ay,az;
	float r1,r3,r5,dr,ds,rv3,bx,by,bz,ECGs;
	float der[NL],ders[NL];
	double grad[6];
	float dpl[3];
	float posi, posj, posk;
	float r2,GRD;
	float tmpdpl;

	// endocardial
	int n0,n1,n2,ni;
	float *surfPOTi,*u1;

	short int nhb, eTime;
	//long temploc;
	int tid=omp_get_thread_num();
	int  myid, numprocs;
	int  namelen;

	//   fprintf(stdout,"BSPitmm !! tid= %d myid= %d numprocs= %d is processor_name= %s,iTime0= %d\n",tid, myid, numprocs, processor_name,iTime0);
	//fflush(stdout);

	short int countijk=0;//by sf-090329***countijk临时记录ijk需要写的次数,gatherijk[20000],
	//float gatherdpl[60000];//by sf-090321***countijk临时记录ijk需要写的次数
	float *endoHnnA_old,*POTi_old,VCG_old[3];//by sf-090402-4
	endoHnnA_old=(float *) malloc(2*NENDO*ND3*4);//by sf-090402-4
	POTi_old=(float *) malloc(NL*4);//by sf-090403-1
	float *epicPOTold;
	epicPOTold=(float *) malloc(Nepic*4);
	for(i=0;i<Nepic;i++) *(epicPOTold+i)=(float)0;
	//double bsptimes1[3]={0.0,0.0,0.0};bsptimes1[0] = clock();


	surfPOTi=(float *) malloc((NL-2)*2*4);
	u1=(float *) malloc((NL-2)*2*4);
	if ((surfPOTi==NULL)||(u1==NULL)) {
		MessageBox(NULL,"Out of memory !",NULL,MB_OK);
		return;// 0;
	}  
	for(ni=0;ni<(NL-2)*2;ni++) {
		*(surfPOTi+ni)=(float)0;
		*(u1+ni)=(float)0;		
	}
	for(ni=0;ni<(NendoB+NendoC);ni++) {
		*(endoHnnA_old+ni)=(float)0;
	}

	// ------- initialization ---------
	for(i=0;i<NL;i++) ders[i]=(float)0; 

	// Save dipole data
	CFile f;
	CFileException e;
	short int index;	


	while (1) {
		idpl = 0;
		asd=(float)0;
		add=(float)0;
		nsum=0;
		rtmax=(float)0;

		for(n=0;n<nPos;n++) {
			*(POTi_old+n)=(float)0;
			der[n]=(float)0;
		}
		for(n=0;n<3;n++) VCG_old[n]=(float)0;//VCG[n]=(float)0;


		//tid=omp_get_thread_num();	
		//if (useGPU==1  && tid==0) gpu_BSPitmm_Malloc(POTi_old,der,endoHnnA_old,surfPOTi);//Comment by SWF (2009-2-7-15)(For:)//by sf-090402-4

		//f.Write(&iTime0,2);//by sf 090329
		//add fibre conduction contribution, iCell

		if (useGPU==1 && tid==0 && GPUnum>0) 
		{
			gpu_BSPitmm_HostToDevice(POTi_old,der,endoHnnA_old,surfPOTi);
		};

		for (nhb=0; nhb<nHB; nhb++) {
			i=iHB[0][nhb];
			j=iHB[1][nhb];
			k=iHB[2][nhb];
			for (ni=0;ni<mxcycle;ni++) {
				eTime=vHB[ni][nhb];
				if (eTime==(short int)(iTime0/3)) {
					compo=(aptcalm(i,j,k,4,iTime0)+90)/nskip/nskip;
					dpl[0]=compo/10;
					dpl[1]=compo/10;
					dpl[2]=compo;
					if (OK_SAV==1) {
						//by sf-090329
						//f.Write(&i,2);
						//f.Write(&j,2);
						//f.Write(&k,2);
						//for (n=0;n<3;n++) {
						//	f.Write(&dpl[n],4);
						//}
						*(gatherallijk[iTime0/3]+countijk)=i;*(gatherallijk[iTime0/3]+countijk+1)=j;*(gatherallijk[iTime0/3]+countijk+2)=k;
						*(gatheralldpl[iTime0/3]+countijk)=dpl[0];*(gatheralldpl[iTime0/3]+countijk+1)=dpl[1];*(gatheralldpl[iTime0/3]+countijk+2)=dpl[2];
						countijk=countijk+3;

						idpl++;
					}

					posi=HRTx0+i*tmswf[0][0]+j*tmswf[0][1]+k*tmswf[0][2];
					posj=HRTy0+i*tmswf[1][0]+j*tmswf[1][1]+k*tmswf[1][2];
					posk=HRTz0+i*tmswf[2][0]+j*tmswf[2][1]+k*tmswf[2][2];
					//  potential distribution generated by
					//  a single dipole in infinite medium
					if (useGPU==1 && tid==0 && GPUnum>0)
					{
						//gpu_BSPitmm_HostToDevice(POTi,der,endoHnnA,surfPOTi);

						gpu_dpl_all(0,posi,posj,posk,nPos,dpl,POTi_old,der,HRTx0,HRTy0,HRTz0,NendoB,NendoC,endoHnnA_old,endoBx,endoBy,endoBz,tmswf,epicPOTold);
						//gpu_dpl_nPos(posi,posj,posk,nPos,dpl,POTi_old,der);
						//gpu_dpl_Nendo(posi,posj,posk,HRTx0,HRTy0,HRTz0,NendoB,0,dpl,endoHnnA_old,endoBx,endoBy,endoBz,tmswf);
						//gpu_dpl_Nendo(posi,posj,posk,HRTx0,HRTy0,HRTz0,NendoC,NendoB,dpl,endoHnnA_old,endoCx,endoCy,endoCz,tmswf);
						//gpu_dpl_nPos_2(posi,posj,posk,dpl);

						//gpu_BSPitmm_DeviceToHost(POTi,der,endoHnnA,surfPOTi);
					}
					else
					{
						///*  //sf	
						for(n=0;n<nPos;n++) {
							ax=*(r[0]+n)-posi;
							ay=*(r[1]+n)-posj;
							az=*(r[2]+n)-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							r5=(float)(r2*r3);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							ds=3*dr/r5;
							rv3=1/r3;
							bx=dpl[0]*rv3-ax*ds;
							by=dpl[1]*rv3-ay*ds;
							bz=dpl[2]*rv3-az*ds;
							*(POTi_old+n)+=dr*rv3;
							*(der+n)+=*(rn[0]+n)*bx+*(rn[1]+n)*by+*(rn[2]+n)*bz;
						}
						//TRACE("\niCell4 %d %d %d %d %f %f",iTime0,i,j,k,compo,*(POTi+94));

						for(n=0;n<NendoB;n++) {
							// ---- measurement location -------
							ax=HRTx0+endoBx[n]*tmswf[0][0]+endoBy[n]*tmswf[0][1]+endoBz[n]*tmswf[0][2]-posi;
							ay=HRTy0+endoBx[n]*tmswf[1][0]+endoBy[n]*tmswf[1][1]+endoBz[n]*tmswf[1][2]-posj;
							az=HRTz0+endoBx[n]*tmswf[2][0]+endoBy[n]*tmswf[2][1]+endoBz[n]*tmswf[2][2]-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							rv3=1/r3;
							*(endoHnnA_old+n)+=dr*rv3;
						}
						for(n=0;n<NendoC;n++) {
							// ---- measurement location -------
							ax=HRTx0+endoCx[n]*tmswf[0][0]+endoCy[n]*tmswf[0][1]+endoCz[n]*tmswf[0][2]-posi;
							ay=HRTy0+endoCx[n]*tmswf[1][0]+endoCy[n]*tmswf[1][1]+endoCz[n]*tmswf[1][2]-posj;
							az=HRTz0+endoCx[n]*tmswf[2][0]+endoCy[n]*tmswf[2][1]+endoCz[n]*tmswf[2][2]-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							rv3=1/r3;
							*(endoHnnA_old+n+NendoB)+=dr*rv3;
						}
						for(n=0;n<(NL-2)*2;n++) {
							// ---- measurement location -------
							n0=*(tnd[0]+n)-1;
							n1=*(tnd[1]+n)-1;
							n2=*(tnd[2]+n)-1;
							ax=(*(r[0]+n0)+*(r[0]+n1)+*(r[0]+n2))/3-posi;
							ay=(*(r[1]+n0)+*(r[1]+n1)+*(r[1]+n2))/3-posj;
							az=(*(r[2]+n0)+*(r[2]+n1)+*(r[2]+n2))/3-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							rv3=1/r3;
							*(surfPOTi+n)+=dr*rv3;
							//Uinf
						}
						//*/   //sf	
					}
				}
			}
		}
		/*
		for (nhb=0; nhb<nttl; nhb++) {
		i=ipttl[0][nhb];
		j=ipttl[1][nhb];
		k=ipttl[2][nhb];
		iCell=*(mapCell[k]+j*NJ+i);
		if (iCell<=1) continue;
		if (iCell>=15) continue;
		temploc=*(locXCT[k]+j*NJ+i);
		eTime=*(mapXCTm[iTime0/mBCL]+temploc);
		if (eTime==(short int)(iTime0/3)) {
		dpl[0]=50;
		dpl[1]=50;
		dpl[2]=50;
		posi=HRTx0+i*tmswf[0][0]+j*tmswf[0][1]+k*tmswf[0][2];
		posj=HRTy0+i*tmswf[1][0]+j*tmswf[1][1]+k*tmswf[1][2];
		posk=HRTz0+i*tmswf[2][0]+j*tmswf[2][1]+k*tmswf[2][2];
		//  potential distribution generated by
		//  a single dipole in infinite medium
		for(n=0;n<nPos;n++) {
		ax=*(r[0]+n)-posi;
		ay=*(r[1]+n)-posj;
		az=*(r[2]+n)-posk;
		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		r5=(float)(r2*r3);
		dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
		ds=3*dr/r5;
		rv3=1/r3;
		bx=dpl[0]*rv3-ax*ds;
		by=dpl[1]*rv3-ay*ds;
		bz=dpl[2]*rv3-az*ds;
		*(POTi+n)+=dr*rv3;
		*(der+n)+=*(rn[0]+n)*bx+*(rn[1]+n)*by+*(rn[2]+n)*bz;
		}
		TRACE("\niCellx %d %d %d %d %f %f",iTime0,i,j,k,compo,*(POTi+94));

		for(n=0;n<NendoB;n++) {
		// ---- measurement location -------
		ax=HRTx0+endoBx[n]*tmswf[0][0]+endoBy[n]*tmswf[0][1]+endoBz[n]*tmswf[0][2]-posi;
		ay=HRTy0+endoBx[n]*tmswf[1][0]+endoBy[n]*tmswf[1][1]+endoBz[n]*tmswf[1][2]-posj;
		az=HRTz0+endoBx[n]*tmswf[2][0]+endoBy[n]*tmswf[2][1]+endoBz[n]*tmswf[2][2]-posk;
		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
		rv3=1/r3;
		*(endoHnnA+n)+=dr*rv3;
		}
		for(n=0;n<NendoC;n++) {
		// ---- measurement location -------
		ax=HRTx0+endoCx[n]*tmswf[0][0]+endoCy[n]*tmswf[0][1]+endoCz[n]*tmswf[0][2]-posi;
		ay=HRTy0+endoCx[n]*tmswf[1][0]+endoCy[n]*tmswf[1][1]+endoCz[n]*tmswf[1][2]-posj;
		az=HRTz0+endoCx[n]*tmswf[2][0]+endoCy[n]*tmswf[2][1]+endoCz[n]*tmswf[2][2]-posk;
		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
		rv3=1/r3;
		*(endoHnnA+n+NendoB)+=dr*rv3;
		}
		for(n=0;n<(NL-2)*2;n++) {
		// ---- measurement location -------
		n0=*(tnd[0]+n)-1;
		n1=*(tnd[1]+n)-1;
		n2=*(tnd[2]+n)-1;
		ax=(*(r[0]+n0)+*(r[0]+n1)+*(r[0]+n2))/3-posi;
		ay=(*(r[1]+n0)+*(r[1]+n1)+*(r[1]+n2))/3-posj;
		az=(*(r[2]+n0)+*(r[2]+n1)+*(r[2]+n2))/3-posk;
		r2=ax*ax+ay*ay+az*az;
		r1=(float)sqrt(r2);
		r3=(float)(r1*r2);
		dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
		rv3=1/r3;
		*(surfPOTi+n)+=dr*rv3;
		//Uinf
		}
		}						
		}
		*/
		//gpu_BSPitmm_HostToDevice(POTi,der,endoHnnA,surfPOTi);
		//printf("Time=%d,\n", iTime0);
		// up, bottom, left, right, front, behind
		for (k=0;k<=NK;k+=nskip) { // i,j // < --> <= August 10,1996
			for (j=0;j<NJ;j+=nskip) {
				for (i=NI;i>-1;i-=nskip) {
					if (k<*(kmin+NI*j+i) || k>*(kmax+NI*j+i)) {
						continue;
					}
					iCell=*(mapCell[k]+NI*j+i);
					// +++++++++ special fiber neglected +++++
					if (iCell<=1) continue;  /*<Comment by ALF> null or SN*/
					if (iCell>=15) continue;  /*<Comment by ALF> out of define*/
					// include fiber conduction
					if((iCell>=3)&&(iCell<=6)) continue; /*<Comment by ALF> not AVN HB BB PKJ*/						
					compo=aptcalm(i,j,k,iCell,iTime0);
					// --------- neighberhood search ---------
					gsum=(float)0;
					for (l=0;l<6;l++) {
						compm=(float)0.0;
						compp=(float)0.0;
						grad[l]=(double)0.0;
						ix=i+iseqx[l];
						iy=j+iseqy[l];
						iz=k+iseqz[l];
						if ((ix>=0)&&(ix<NI)&&(iy>=0)&&(iy<NJ)&&(iz>=0)&&(iz<NK)) {
							icell=*(mapCell[iz]+iy*NI+ix);
							if ((icell>1)&&(icell<15)&&((icell<3)||(icell>6)))  {
								compm=aptcalm(ix,iy,iz,icell,iTime0);
								//if (iTime0 ==3 && compm != -90.) TRACE("\nB %d %d %d %f",ix,iy,iz, compm);
								grad[l]+=compm-compo;
							} 
						}

						l6=l+6;  /*<Comment by ALF> opposite one*/
						jx=i+iseqx[l6];
						jy=j+iseqy[l6];
						jz=k+iseqz[l6];
						if ((jx>=0)&&(jx<NI)&&(jy>=0)&&(jy<NJ)&&(jz>=0)&&(jz<NK)) {
							jcell=*(mapCell[jz]+jy*NI+jx);
							if ((jcell>1)&&(jcell<15)&&((jcell<3)||(jcell>6)))  {
								compp=aptcalm(jx,jy,jz,jcell,iTime0);
								grad[l]+=compo-compp;
							} 
						}
					}

					for (l=0;l<6;l++)
						gsum+=(float)fabs((double)grad[l]);
					if (gsum==0) continue; 
					// close dpl file					
					//  dipole number --> nsum; position-->ipos
					for (n=0;n<3;n++) {
						dpl[n]=(float)0;
						for (short int m=0;m<6;m++) 
							dpl[n]+=tmswf[n][m]*grad[m];
						// -- take conductivity factor into consideration --
						dpl[n]=dpl[n]*(*(iparm+NPARM*(iCell-1)+12))/(100);						
						// f.Write(&dpl[n],4);
						// *(dplm[n]+idpl) = dpl[n];
						// >>>>> moved to an independent loop below >>>>
						// VCG[n]+=dpl[n];
					}								
					// >>>>>>>>>> aniso >>>>>>>>
					tmpdpl=dpl[0];
					if (ANISO==1 && icell==7) {
						anfct(i,j,k,dpl);
					}
					// if (tmpdpl-dpl[0]>0.0001 || tmpdpl-dpl[0]<-0.0001) 
					//	TRACE("\ndpl %2d %2d %2d %f %f",i+1,j+1,k+1, tmpdpl, dpl[0]);
					if (OK_SAV==1) {
						//by sf-090329
						//f.Write(&i,2);
						//f.Write(&j,2);
						//f.Write(&k,2);
						//for (n=0;n<3;n++) {
						//	f.Write(&dpl[n],4);
						//}
						*(gatherallijk[iTime0/3]+countijk)=i;*(gatherallijk[iTime0/3]+countijk+1)=j;*(gatherallijk[iTime0/3]+countijk+2)=k;
						*(gatheralldpl[iTime0/3]+countijk)=dpl[0];*(gatheralldpl[iTime0/3]+countijk+1)=dpl[1];*(gatheralldpl[iTime0/3]+countijk+2)=dpl[2];
						countijk=countijk+3;

					}
					for (n=0;n<3;n++) {
						VCG_old[n]+=dpl[n];//VCG[n]+=dpl[n];
					}
					idpl++;
					// <<<<<<<<<< aniso <<<<<<<<
					posi=HRTx0+i*tmswf[0][0]+j*tmswf[0][1]+k*tmswf[0][2];
					posj=HRTy0+i*tmswf[1][0]+j*tmswf[1][1]+k*tmswf[1][2];
					posk=HRTz0+i*tmswf[2][0]+j*tmswf[2][1]+k*tmswf[2][2];
					//  potential distribution generated by
					//  a single dipole in infinite medium
					//------------ 2009-2-4-16 BY SWF---------
					// comment:
					//printf("nPos*,itime0=%d", iTime0);



					if (useGPU==1 && tid==0 && GPUnum>0)
					{
						//	gpu_freetransdata();
						//gpu_transdata(tnd,r,rn,endoBx,endoBy,endoBz,endoCx,endoCy,endoCz,tmswf);
						//gpu_BSPitmm_HostToDevice(POTi,der,endoHnnA,surfPOTi);
						gpu_dpl_all(1,posi,posj,posk,nPos,dpl,POTi_old,der,HRTx0,HRTy0,HRTz0,NendoB,NendoC,endoHnnA_old,endoBx,endoBy,endoBz,tmswf,epicPOTold);
						//gpu_dpl_nPos(posi,posj,posk,nPos,dpl,POTi_old,der);
						//gpu_dpl_Nendo(posi,posj,posk,HRTx0,HRTy0,HRTz0,NendoB,0,dpl,endoHnnA_old,endoBx,endoBy,endoBz,tmswf);
						//gpu_dpl_Nendo(posi,posj,posk,HRTx0,HRTy0,HRTz0,NendoC,NendoB,dpl,endoHnnA_old,endoCx,endoCy,endoCz,tmswf);
						//gpu_dpl_nPos_2(posi,posj,posk,dpl);
						//gpu_dpl_Nepic(posi,posj,posk,HRTx0,HRTy0,HRTz0,dpl,tmswf,epicPOTold);

						//gpu_BSPitmm_DeviceToHost(POTi,der,endoHnnA,surfPOTi);
						///*							//printf("$");
					}

					else
					{

						for(n=0;n<nPos;n++) {
							ax=*(r[0]+n)-posi;
							ay=*(r[1]+n)-posj;
							az=*(r[2]+n)-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							r5=(float)(r2*r3);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							ds=3*dr/r5;
							rv3=1/r3;
							bx=dpl[0]*rv3-ax*ds;
							by=dpl[1]*rv3-ay*ds;
							bz=dpl[2]*rv3-az*ds;
							*(POTi_old+n)+=dr*rv3;
							*(der+n)+=*(rn[0]+n)*bx+*(rn[1]+n)*by+*(rn[2]+n)*bz;
						}

						//  endocadial potential distribution generated by
						//  a single dipole in infinite medium
						for(n=0;n<NendoB;n++) {
							// ---- measurement location -------
							ax=HRTx0+endoBx[n]*tmswf[0][0]+endoBy[n]*tmswf[0][1]+endoBz[n]*tmswf[0][2]-posi;
							ay=HRTy0+endoBx[n]*tmswf[1][0]+endoBy[n]*tmswf[1][1]+endoBz[n]*tmswf[1][2]-posj;
							az=HRTz0+endoBx[n]*tmswf[2][0]+endoBy[n]*tmswf[2][1]+endoBz[n]*tmswf[2][2]-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							rv3=1/r3;
							*(endoHnnA_old+n)+=dr*rv3;
						}



						for(n=0;n<NendoC;n++) {
							// ---- measurement location -------
							ax=HRTx0+endoCx[n]*tmswf[0][0]+endoCy[n]*tmswf[0][1]+endoCz[n]*tmswf[0][2]-posi;
							ay=HRTy0+endoCx[n]*tmswf[1][0]+endoCy[n]*tmswf[1][1]+endoCz[n]*tmswf[1][2]-posj;
							az=HRTz0+endoCx[n]*tmswf[2][0]+endoCy[n]*tmswf[2][1]+endoCz[n]*tmswf[2][2]-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							rv3=1/r3;
							*(endoHnnA_old+n+NendoB)+=dr*rv3;
						}



						//-------------------- modified by ALF at 2008-8-19 begin -------------------->
						//add: epicardial potential distribution generated by
						//  a single dipole in infinite medium
						for (n=0; n<Nepic; ++n) {
							ax=HRTx0+epicX[n]*tmswf[0][0]+epicY[n]*tmswf[0][1]+epicZ[n]*tmswf[0][2]-posi;
							ay=HRTy0+epicX[n]*tmswf[1][0]+epicY[n]*tmswf[1][1]+epicZ[n]*tmswf[1][2]-posj;
							az=HRTz0+epicX[n]*tmswf[2][0]+epicY[n]*tmswf[2][1]+epicZ[n]*tmswf[2][2]-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							rv3=1/r3;
							*(epicPOT+n)+=dr*rv3;
						}
						//-------------------- modified by ALF at 2008-8-19 end --------------------< 
						for(n=0;n<(NL-2)*2;n++) {
							// ---- measurement location -------
							n0=*(tnd[0]+n)-1;
							n1=*(tnd[1]+n)-1;
							n2=*(tnd[2]+n)-1;
							ax=(*(r[0]+n0)+*(r[0]+n1)+*(r[0]+n2))/3-posi;
							ay=(*(r[1]+n0)+*(r[1]+n1)+*(r[1]+n2))/3-posj;
							az=(*(r[2]+n0)+*(r[2]+n1)+*(r[2]+n2))/3-posk;
							r2=ax*ax+ay*ay+az*az;
							r1=(float)sqrt(r2);
							r3=(float)(r1*r2);
							dr=dpl[0]*ax+dpl[1]*ay+dpl[2]*az;
							rv3=1/r3;
							*(surfPOTi+n)+=dr*rv3;
							//Uinf
						}

					};////test  sf
					//*/			
					//------------ 2009-2-4-16 BY SWF---------

				}
			}
		}
		if (useGPU==1 && tid==0 && GPUnum>0)
		{
			gpu_BSPitmm_DeviceToHost(epicPOTold,POTi_old,der,endoHnnA_old,surfPOTi);
			for(i=0;i<Nepic;i++) *(epicPOT+i)=*(epicPOTold+i);
		}
		//bsptimes1[1]   =   clock();
		//if (iTime0<1800  )   //sf
		//    	{float dd=0,pp=0;
		//		for(int ff=0;ff<NendoB;ff++)
		//		{
		//			dd+=*(endoHnnA+ff);
		//			//pp+=*(POTi+ff);
		//		}
		//		FILE *fptime;
		//		fptime=fopen(dataPath+"data.txt","a")  ;
		//		fprintf(fptime,"iTime0=%d,endoHnnA=%f,,\n",iTime0,dd);
		//		printf("iTime0=%d,endoHnnA=%f,,\n",iTime0,dd);
		//		fclose(fptime);
		//		};


		//---- next Step -----
		GRD=0; // ?  April 29, 1996
		for(i=0;i<3;i++)
			GRD+=(bufVCG[1][i]-VCG_old[i])*(bufVCG[1][i]-VCG_old[i]);//GRD+=(bufVCG[1][i]-VCG[i])*(bufVCG[1][i]-VCG[i]);
		intvl=iTime0-itbuf;
		GRD=sqrt(GRD)/intvl;
		GRD=100*GRD/939.513;
		/*
		i=0;
		for (ni=0;ni<mxcycle;ni++) {
		eTime=vHB[ni][0];
		if ((eTime-iTime0/3)<4*ND) {
		i=1;
		}	
		}
		if (i==1) {
		nextStep=3*ND;
		break;
		}

		if (GRD>10*ND) {   
		if((bufGRD<2)&&(intvl>3)) {
		iTime0=itbuf+3;
		continue;
		}
		nextStep=3*ND;
		//nextStep=ND;
		break;
		} 
		if (GRD>5*ND) { 
		nextStep=6*ND;  // 9 --> 6 August 11, 1996
		//nextStep=2*ND;  // 9 --> 6 August 11, 1996
		break;
		} 
		nextStep=12*ND;  // 21 --> 12 August 11, 1996
		*/
		nextStep=3*ND;  // 21 --> 12 August 11, 1996		
		break;		
	}

	itbuf=iTime0;
	bufGRD=(float)GRD;
	// ---- the same value with the previous two ? --
	for (n=0;n<3;n++) {
		if ((VCG_old[n]!=bufVCG[0][n])||(VCG_old[n]!=bufVCG[1][n])) {//if ((VCG[n]!=bufVCG[0][n])||(VCG[n]!=bufVCG[1][n])) {
			n=-1;
			break;
		}
	}
	//if (n != -1) {
	//	answer='s';
	//	answer='d';
	//	//return;
	//}
	//else
	//{
	answer='d';
	for (n=0;n<3;n++) {
		bufVCG[0][n]=bufVCG[1][n];
		bufVCG[1][n]=VCG_old[n];//bufVCG[1][n]=VCG[n];
	}


	// --- boundary condition into Account-------
	ECGs=(float)0;
	for(j=0;j<nPos;j++) ECGs+=*(POTi_old+j);
	ECGs*=alp;
	for(j=0;j<nPos;j++) {
		*(ders+j)=(float)0;
		for(k=0;k<nPos;k++) *(ders+j)+=*(aw[j]+k)*(*(der+k)); // aw : j,k or k,j ?
		*(POTi_old+j)+=-*(ders+j)-*(bw+j)*ECGs;
	}

	// body surface triangle
	float sum, tmp, triarea, sumarea;
	for (j=0; j<(NL-2)*2;j++) {
		sum=0.0;
		for(k=0;k<(nPos-2)*2;k++) {
			tmp=*(surfPOTi+k);
			sum+=*(hnn+j*(nPos-2)*2+k) * tmp;
		}	
		*(u1+j)=sum;			
	}	
	triarea=0.0;
	sumarea=0.0;
	for(n=0;n<(NL-2)*2;n++) {
		// ---- measurement location -------
		n0=*(tnd[0]+n)-1;
		n1=*(tnd[1]+n)-1;
		n2=*(tnd[2]+n)-1;
		ax=(*(r[0]+n0)-*(r[0]+n1));
		ay=(*(r[1]+n0)-*(r[1]+n1));
		az=(*(r[2]+n0)-*(r[2]+n1));
		bx=(*(r[0]+n0)-*(r[0]+n2));
		by=(*(r[1]+n0)-*(r[1]+n2));
		bz=(*(r[2]+n0)-*(r[2]+n2));
		tmp=(ax*by-bx*ay)*(ax*by-bx*ay)+(ax*bz-bx*az)*(ax*bz-bx*az)+(az*by-bz*ay)*(az*by-bz*ay);
		tmp=0.5*sqrt(tmp);
		triarea+=*(u1+n)*tmp;
		sumarea+=tmp;
	}
	for (n=0; n<NendoB;n++) {
		sum=0.0;
		for(k=0;k<(nPos-2)*2;k++) {
			tmp=*(u1+k);
			sum +=*(endoHnnB+n*(nPos-2)*2+k) * tmp;
		}	
		*(endoHnnA_old+n)+=sum-triarea/sumarea;		
	}
	for (n=0; n<NendoC;n++) {
		sum=0.0;
		for(k=0;k<(nPos-2)*2;k++) {
			tmp=*(u1+k);
			sum +=*(endoHnnC+n*(nPos-2)*2+k) * tmp;
		}	
		*(endoHnnA_old+n+NendoB)+=sum-triarea/sumarea;		
	}
	//}//sf-090402-5 if (n != -1) {
	//-------------------- modified by ALF at 2008-8-19 begin -------------------->
	//add
	for (n=0; n<Nepic;n++) {
		sum=0.0;
		for(k=0;k<(nPos-2)*2;k++) {
			tmp=*(u1+k);
			sum +=*(epicHnn+n*(nPos-2)*2+k) * tmp;
		}	
		*(epicPOT+n)+=sum-triarea/sumarea;		
	}
	//-------------------- modified by ALF at 2008-8-19 end --------------------< 
	//by sf 090329
	/*
	#pragma omp critical
	{//critical--begin
	if (OK_SAV==1) {		
	CFile f2;
	CFileException e2;
	//short int index2;	
	//index2=filepath.FindOneOf(".");
	//filepath.SetAt(index2+1,'d');
	//filepath.SetAt(index2+2,'p');
	//filepath.SetAt(index2+3,'n');
	if (iTime0 > 3) {
	if (!f2.Open(dataPath+"tour.dpn ",CFile::modeReadWrite, &e2 )) { 
	f2.Open(dataPath+"tour.dpn ",CFile::modeCreate|CFile::modeReadWrite, &e2 ); 
	}
	f2.SeekToEnd();
	} else {
	if (!f2.Open( dataPath+"tour.dpn ", CFile::modeCreate | CFile::modeWrite, &e2 )) {
	#ifdef _DEBUG
	afxDump << "File could not be opened " << e2.m_cause << "\n";
	#endif
	}
	}
	f2.Write(&iTime0,2);
	f2.Write(&idpl,2);
	f2.Close();
	}

	if (OK_SAV==1) {
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'d');
	//filepath.SetAt(index+2,'p');
	//filepath.SetAt(index+3,'l');

	if (iTime0 > 3) {
	if (!f.Open(dataPath+"tour.dpl ",CFile::modeReadWrite, &e )) { 
	f.Open(dataPath+"tour.dpl ",CFile::modeCreate|CFile::modeReadWrite, &e ); 
	}
	f.SeekToEnd();
	} else {
	if (!f.Open(dataPath+"tour.dpl ", CFile::modeCreate | CFile::modeWrite, &e )) {
	#ifdef _DEBUG
	afxDump << "File could not be opened " << e.m_cause << "\n";
	#endif
	}
	}
	}

	f.Write(&iTime0,2);//from line 4070
	//f.Write(gatherijk,2*countijk);
	//f.Write(gatherdpl,4*countijk);
	for(j=0;j<countijk;j=j+3)
	{			
	f.Write(gatherijk+j,2*3);
	//f.Write(gatherijk+j+1,2);
	//f.Write(gatherijk+j+2,2);
	f.Write(gatherdpl+j,4*3);
	//f.Write(gatherdpl+j+1,4);
	//f.Write(gatherdpl+j+2,4);
	}
	if (OK_SAV==1) {
	f.Close();
	}
	}//critical--end
	*/
	//by sf  090408 for dpl[] 
	// int tmpiTime=iTime0/3;
	//  gatherallijk[tmpiTime]   =   (int   *)   malloc(   countijk*sizeof(int)   ); 
	//  gatheralldpl[tmpiTime]   =   (float   *)   malloc(   countijk*sizeof(float)   ); 
	//  *(countallijk+tmpiTime)=countijk;
	// for(j=0;j<countijk;j=j+1)
	//{			
	//	*(gatherallijk[tmpiTime]+j)=gatherijk[j];
	//	*(gatheralldpl[tmpiTime]+j)=gatherdpl[j];
	// }
	// if(iTime0==27)
	//{
	//	printf("27bcasti=%d,myid=%d,ijk=%d,%d,g=%d,%d\n",iTime0,myid,*(gatherallijk[iTime0/3]),*(gatherallijk[iTime0/3]+1),gatherijk[0],gatherijk[1]);
	//	printf("27bcasti=%d,myid=%d,dpl=%f,%f,%f,%f\n",iTime0,myid,*(gatheralldpl[iTime0/3]),*(gatheralldpl[iTime0/3]+1),gatherdpl[0],gatherdpl[1]);
	//};

	//  by sf 090401 BSPMcal if begin
	short int nTimeStep_old=iTime0/3;//nTimeStep=nTimeStep+1;
	if ((answer!='s')||(nTimeStep_old<=1)) {
		int n = 0;

		for(n=0;n<nPos;n++) {
			*(POT[n]+nTimeStep_old)=*(POTi_old+n);
		}
		// add endocardial potential 
		//printf("iTime=%d,tid=%d,tnum=%d,nTimeStep_old=%d\n",iTime0,omp_get_thread_num(),omp_get_num_threads(),nTimeStep_old);
		for(n=0;n<2*NENDO*ND3;n++) {
			*(endoPOT[nTimeStep_old-1]+n)=*(endoHnnA_old+n);
		}
		if(iTime0<=nsnrt) {
			//nVCG_old++;//nVCG=nVCG+1;nVCG-->nTimeStep
			for(n=0;n<3;n++) {
				*(VCGs[n]+nTimeStep_old)=VCG_old[n]/ND;	//*(VCGs[n]+nVCG)=VCG[n]/ND;
			}
		}
	}
	//bsptimes1[2]   =   clock();
	//printf("%f,%f,bsptimes1[1-0]-bsptimes1[2-1] tid=%d,iTime0=%d\n",(bsptimes1[1]-bsptimes1[0])/CLK_TCK,(bsptimes1[2]-bsptimes1[1])/CLK_TCK,tid,iTime0);
	free(endoHnnA_old);free(POTi_old);//  by sf 090402-3
	free(epicPOTold);
	//  by sf 090401 if  end

	free(u1);	
	free(surfPOTi);	
}

// *********** action potential calculation *******
float aptcalm(short int i0,short int j0,short int k0,short int iCell0,short int iTime1) {      
	short int istp,irsd,lacl,lacl1,iext;
	float ACTval;

	// ++++ resting potential +++++
	ACTval=(float)(*(iparm+NPARM*(iCell0-1)+6));
	istp=(short int)(iTime1/3); // each step has 3 time slots

	//rdXCTm(istp,i0,j0,k0); //by sf-090401 子程序取消,把该代码贴入,避免idltt和idltc全局变量在OpenMP由于共享出错// get idltt = istp - ncyc, the current step in current cycle
	//idlttold-->idltt  idltcold-->idltc
	short int ncyc,n1cyc;
	short int i00,j00,k00,icc,idlttold,idltcold;
	i00=i0;j00=j0;k00=k0;icc=istp;
	idlttold=INFTIME;  /*<Comment by ALF> period between 2 continuous excitation*/
	idltcold=0;		/*<Comment by ALF> delta of 2 periods */
	short int n;
	long locxct;

	locxct=*(locXCT[k00]+j00*NI+i00);
	if(locxct<0)
	{//return;
	}
	else
	{
		for(n=NCYCL-1;n>=0;n--) {
			ncyc=*(mapXCTm[n]+locxct);
			if (icc>=ncyc) { 
				idlttold=icc-ncyc; 
				break; 
			} 
		}
		if ((n<=0)||(n>=NCYCL-1))
		{//return;
		}
		else
		{
			n1cyc=*(mapXCTm[n+1]+locxct);
			if (n1cyc==INFTIME)
			{
			}
			else
			{
				idltcold=n1cyc-ncyc-ncyc+*(mapXCTm[n-1]+locxct);
			};
		};
	};
	//rdXCTm(istp,i0,j0,k0);---end --by sf

	if (idlttold==INFTIME) { // ACTval=-90 situation
		return ACTval; 
	}
	irsd=iTime1-istp*3;
	idlttold=idlttold*3+irsd;

	// iext=*(mapACT[k0]+j0*NI+i0)+idltc * 3 * *(iparm+(iCell0-1)*NPARM+10)/100;
	iext=*(mapAPD[k0]+j0*NI+i0)+idltcold * 3 * *(iparm+(iCell0-1)*NPARM+10)/100;
	lacl=la0123[iCell0-1]+iext;
	if(idlttold>lacl) return ACTval;
	lacl1=la012[iCell0-1]+iext;
	//TRACE("\naptcalm %d %d %d %d %d %d",idltt,lacl1,la012[iCell0-1],iext,lacl,*(mapAPD[k0]+j0*NI+i0));
	if(idlttold>lacl1) {
		idlttold-=iext;
		ACTval=*(ydata[iCell0-1]+idlttold);
		return ACTval;
	}
	if(idlttold>la012[iCell0-1]) idlttold=la012[iCell0-1]; 
	ACTval=*(ydata[iCell0-1]+idlttold);
	return ACTval;
	// --- atrial REPolarization ignored ---- 
	//      if(iCell0==2) {
	//         ACTval=(float)(*(iparm+1*NPARM+7));
	//         return ACTval;
	//         }
	// +++ la012, time to the end of phase 2; +++++
	// +++ la0123,time to the end of phase 3  +++++
}


// ECG calculation 
void ECGcal(void) {
	float ECG[12],ECGr,ECGl,ECGf;
	short int iECG[12],i,j;
	float wilson;
	float eff=(float)26.5730/ND; 

	// Save ecg data

	CFile f;
	CFileException e;
	//short int index;
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'e');
	//filepath.SetAt(index+2,'c');
	//filepath.SetAt(index+3,'g');

	if (!f.Open( dataPath+"tour.ecg ", CFile::modeCreate | CFile::modeWrite, &e )) {
#ifdef _DEBUG
		afxDump << "File could not be opened " << e.m_cause << "\n";
#endif
	}

	//FILE *fptime;//sf
	//fptime=fopen(dataPath+"ecg-int-gpu.txt","w")  ;//sf
	//fprintf(fptime,"%d\n",nTimeStep);//sf

	f.Write(&nTimeStep,2);
	for (j=1;j<=nTimeStep;j++) 
	{f.Write(iStep+j,2);
	//fprintf(fptime,"%d\n",*(iStep+j));//sf
	}
	//fclose(fptime);//sf
	//fptime=fopen(dataPath+"ecg-float-gpu.txt","w")  ;//sf
	// Compute ECG 
	for (i=1;i<=nTimeStep;i++) {
		//ECGr=*(POT[nv[0]]+i);
		//ECGl=*(POT[nv[1]]+i);
		//ECGf=*(POT[nv[2]]+i);
		ECGr=*(POT[nv[0]]+i);
		ECGl=*(POT[nv[1]]+i);
		ECGf=*(POT[nv[2]]+i);
		wilson=(ECGr+ECGl+ECGf)/3;
		ECG[0]=*(POT[94]+i)-wilson;
		ECG[1]=*(POT[96]+i)-wilson;
		ECG[2]=*(POT[117]+i)-wilson;
		ECG[3]=*(POT[138]+i)-wilson;
		ECG[4]=(*(POT[139]+i)/2+*(POT[140]+i)/2)-wilson;
		ECG[5]=*(POT[141]+i)-wilson;
		ECG[6]=ECGl-ECGr;
		ECG[7]=ECGf-ECGr;
		ECG[8]=ECGf-ECGl;
		ECG[9]=(ECGr-wilson)*3/2;
		ECG[10]=(ECGl-wilson)*3/2;
		ECG[11]=(ECGf-wilson)*3/2;
		iECG[0]=(short int)(eff*ECG[0]);
		iECG[1]=(short int)(eff*ECG[1]);
		iECG[2]=(short int)(eff*ECG[2]);
		iECG[3]=(short int)(eff*ECG[3]);
		iECG[4]=(short int)(eff*ECG[4]);
		iECG[5]=(short int)(eff*ECG[5]);
		iECG[6]=(short int)(eff*ECG[6]);
		iECG[7]=(short int)(eff*ECG[7]);
		iECG[8]=(short int)(eff*ECG[8]);
		iECG[9]=(short int)(eff*ECG[9]);
		iECG[10]=(short int)(eff*ECG[10]);
		iECG[11]=(short int)(eff*ECG[11]);
		for (j=0;j<12;j++) {
			f.Write(&iECG[j],2);
			//fprintf(fptime,"%d\n",iECG[j]);//sf

		}

		//TRACE("\n %3d %5d %f %f %f %f ",i, iECG[6], ECG[6], ECGl,ECGr,ECGf);
	}
	f.Close();

	//fclose(fptime);//sf
}


//-------------------- modified by sf at 2008-4-27 begin -------------------->
//modified
int BSPitmmcount(short int iTime0) {
	//void BSPitmm(short int iTime0, short int **tnd,float *hnn, float *endoHnnA, float *endoHnnB, float *endoHnnC, float *epicHnn, float *epicPOT) {
	//ASSERT(epicHnn != NULL);
	//-------------------- modified by ALF at 2008-8-19 end --------------------< 
	float aptcalm(short int,short int,short int,short int,short int);
	void anfct(short int i, short int j, short int k, float v[3]);
	int loopcount=0;
	char iCell;
	const short int OK_SAV=1; 
	short int iseqx[12]={ -1, 0, 0, 1, 1, 0, 1, 0, 0,-1,-1, 0 };
	short int iseqy[12]={  0,-1, 0,-1, 0, 1, 0, 1, 0, 1, 0,-1 };
	short int iseqz[12]={  0, 0,-1, 0,-1,-1, 0, 0, 1, 0, 1, 1 };
	short int nskip=2;
	short int i,j,k,l,ix,iy,iz,icell,l6,jx,jy,jz,jcell;

	int nsum,n;
	int intvl;
	int idpl;

	float asd,add,rtmax,gsum,compm,compp,compo,ax,ay,az;
	float r1,r3,r5,dr,ds,rv3,bx,by,bz,ECGs;
	//float der[NL],ders[NL];
	double grad[6];
	//float dpl[3];
	float posi, posj, posk;
	float r2,GRD;
	float tmpdpl;

	// endocardial
	int n0,n1,n2,ni;
	//float *surfPOTi,*u1;

	short int nhb, eTime;
	short int index;	

	for (nhb=0; nhb<nHB; nhb++) {
		i=iHB[0][nhb];
		j=iHB[1][nhb];
		k=iHB[2][nhb];
		for (ni=0;ni<mxcycle;ni++) {
			eTime=vHB[ni][nhb];
			if (eTime==(short int)(iTime0/3)) {
				loopcount++;

			}
		}
	}

	for (k=0;k<=NK;k+=nskip) { // i,j // < --> <= August 10,1996
		for (j=0;j<NJ;j+=nskip) {
			for (i=NI;i>-1;i-=nskip) {
				if (k<*(kmin+NI*j+i) || k>*(kmax+NI*j+i)) {
					continue;
				}
				iCell=*(mapCell[k]+NI*j+i);
				// +++++++++ special fiber neglected +++++
				if (iCell<=1) continue;  /*<Comment by ALF> null or SN*/
				if (iCell>=15) continue;  /*<Comment by ALF> out of define*/
				// include fiber conduction
				if((iCell>=3)&&(iCell<=6)) continue; /*<Comment by ALF> not AVN HB BB PKJ*/						
				compo=aptcalm(i,j,k,iCell,iTime0);
				// --------- neighberhood search ---------
				gsum=(float)0;
				for (l=0;l<6;l++) {
					compm=(float)0.0;
					compp=(float)0.0;
					grad[l]=(double)0.0;
					ix=i+iseqx[l];
					iy=j+iseqy[l];
					iz=k+iseqz[l];
					if ((ix>=0)&&(ix<NI)&&(iy>=0)&&(iy<NJ)&&(iz>=0)&&(iz<NK)) {
						icell=*(mapCell[iz]+iy*NI+ix);
						if ((icell>1)&&(icell<15)&&((icell<3)||(icell>6)))  {
							compm=aptcalm(ix,iy,iz,icell,iTime0);
							//if (iTime0 ==3 && compm != -90.) TRACE("\nB %d %d %d %f",ix,iy,iz, compm);
							grad[l]+=compm-compo;
						} 
					}

					l6=l+6;  /*<Comment by ALF> opposite one*/
					jx=i+iseqx[l6];
					jy=j+iseqy[l6];
					jz=k+iseqz[l6];
					if ((jx>=0)&&(jx<NI)&&(jy>=0)&&(jy<NJ)&&(jz>=0)&&(jz<NK)) {
						jcell=*(mapCell[jz]+jy*NI+jx);
						if ((jcell>1)&&(jcell<15)&&((jcell<3)||(jcell>6)))  {
							compp=aptcalm(jx,jy,jz,jcell,iTime0);
							grad[l]+=compo-compp;
						} 
					}
				}

				for (l=0;l<6;l++)
					gsum+=(float)fabs((double)grad[l]);
				if (gsum==0) continue; 
				loopcount++;
			}
		}
	}
	return loopcount;
}

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
	(cudaFree(d_tm));
	(cudaFree(d_endoBx));(cudaFree(d_endoBy));(cudaFree(d_endoBz));
	(cudaFree(d_endoCx));(cudaFree(d_endoCy));(cudaFree(d_endoCz));
	(cudaFree(d_r));
	(cudaFree(d_rn));
	(cudaFree(d_tnd));


}

//int main(int argc, char** argv)
extern "C" short int cudamain(int argc, char** argv)
{//int i;
	fprintf(stdout, "before \n");
	fflush(stdout);
	if(!InitCUDA()) {
		return 0;
	}
	fflush(stdout);
	int count = 0;
	short int GPUnumber;
	cudaGetDeviceCount(&count);
	GPUnumber=count; 
	//hpc(argc, argv);
	printf("CUDA is OK=%d\n",GPUnumber);
	return GPUnumber;
	//for(i=0;i<3;i++) 
	// { 
	//	 //(cudaFree(d_r[i]));(cudaFree(d_rn[i]));
	//	(cudaFree(d_tnd[i]))
	//};

	/*	gpu_freetransdata();//These function should be called by one process on one PC
	(cudaFree(d_POTi));(cudaFree(d_der));
	(cudaFree(d_endoHnnA));(cudaFree(d_surfPOTi));
	CUT_EXIT(argc, argv);
	*/
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
		( cudaMalloc((void**) &d_r, sizeof(float) * NL*3));
		( cudaMemcpy(d_r, cg_r, sizeof(float) * NL*3, cudaMemcpyHostToDevice));
		( cudaMalloc((void**) &d_rn, sizeof(float) * NL*3));
		( cudaMemcpy(d_rn, cg_rn, sizeof(float) * NL*3, cudaMemcpyHostToDevice));

		short int cg_tnd[(NL-2)*2*3];
		for(i=0;i<3;i++)
			for(j=0;j<(NL-2)*2;j++)
			{
				cg_tnd[i*(NL-2)*2+j]=*(g_tnd[i]+j);

			}
			( cudaMalloc((void**) &d_tnd, sizeof(short int) * (NL-2)*2*3));
			( cudaMemcpy(d_tnd, cg_tnd, sizeof(short int) * (NL-2)*2*3, cudaMemcpyHostToDevice));

			//for(i=0;i<3;i++) 
			//{
			// //( cudaMalloc((void**) &d_r[i], sizeof(float) * NL));
			// //( cudaMemcpy((d_r[i]), (g_r[i]), sizeof(float) * NL, cudaMemcpyHostToDevice));
			// //( cudaMalloc((void**) &d_rn[i], sizeof(float) * NL));
			// //( cudaMemcpy((d_rn[i]), (g_rn[i]), sizeof(float) * NL, cudaMemcpyHostToDevice));
			// ( cudaMalloc((void**) &d_tnd[i], sizeof(short int) * (NL-2)*2));
			// ( cudaMemcpy((d_tnd[i]), (g_tnd[i]), sizeof(short int) * (NL-2)*2, cudaMemcpyHostToDevice));
			//};

			float cg_tm[3*6];
			for(i=0;i<3;i++)
				for(j=0;j<6;j++)
				{
					cg_tm[i*6+j]=*(g_tm[i]+j);
				}
				( cudaMalloc((void**) &d_tm, sizeof(float) * 3 * 6));
				( cudaMemcpy(d_tm, cg_tm, (sizeof(float) * 3 * 6), cudaMemcpyHostToDevice));

				( cudaMalloc((void**) &d_epicX, sizeof(short int) * Nepic));
				( cudaMalloc((void**) &d_epicY, sizeof(short int) * Nepic));
				( cudaMalloc((void**) &d_epicZ, sizeof(short int) * Nepic));
				( cudaMemcpy((d_epicX),(g_epicX) , (sizeof(short int) * Nepic), cudaMemcpyHostToDevice));
				( cudaMemcpy(d_epicY,g_epicY , sizeof(short int) * Nepic, cudaMemcpyHostToDevice));
				( cudaMemcpy(d_epicZ, g_epicZ, sizeof(short int) * Nepic, cudaMemcpyHostToDevice));



				( cudaMalloc((void**) &d_endoBx, sizeof(short int) * NENDO*ND3));
				( cudaMalloc((void**) &d_endoBy, sizeof(short int) * NENDO*ND3));
				( cudaMalloc((void**) &d_endoBz, sizeof(short int) * NENDO*ND3));

				( cudaMalloc((void**) &d_endoCx, sizeof(short int) * NENDO*ND3));
				( cudaMalloc((void**) &d_endoCy, sizeof(short int) * NENDO*ND3));
				( cudaMalloc((void**) &d_endoCz, sizeof(short int) * NENDO*ND3));


				( cudaMemcpy((d_endoBx),(g_endoBx) , (sizeof(short int) * NENDO*ND3), cudaMemcpyHostToDevice));
				( cudaMemcpy(d_endoBy,g_endoBy , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
				( cudaMemcpy(d_endoBz, g_endoBz, sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));

				( cudaMemcpy(d_endoCx,g_endoCx , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
				( cudaMemcpy(d_endoCy,g_endoCy , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
				( cudaMemcpy(d_endoCz,g_endoCz , sizeof(short int) * NENDO*ND3, cudaMemcpyHostToDevice));
				//申请计算用数据空间,这样只要一次申请
				( cudaMalloc((void**) &d_epicPOTold, sizeof(float) * Nepic));
				( cudaMalloc((void**) &d_POTi, sizeof(float) * NL));
				( cudaMalloc((void**) &d_der, sizeof(float) * NL));
				( cudaMalloc((void**) &d_endoHnnA, sizeof(float) * 2*NENDO*ND3));
				( cudaMalloc((void**) &d_surfPOTi, sizeof(float) * (NL-2)*2));

}
//extern "C" void gpu_BSPitmm_Malloc(float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi)
//{
//  ( cudaMalloc((void**) &d_epicPOTold, sizeof(float) * Nepic));
//  ( cudaMalloc((void**) &d_POTi, sizeof(float) * NL));
//  ( cudaMalloc((void**) &d_der, sizeof(float) * NL));
//  ( cudaMalloc((void**) &d_endoHnnA, sizeof(float) * 2*NENDO*ND3));
//  ( cudaMalloc((void**) &d_surfPOTi, sizeof(float) * (NL-2)*2));
//}

extern "C" void gpu_BSPitmm_HostToDevice(float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi)
{
	cudaMemset(d_epicPOTold, 0, sizeof(float) * Nepic);
	cudaMemset(d_POTi, 0, sizeof(float) * NL);
	cudaMemset(d_der, 0,  sizeof(float) * NL);
	cudaMemset(d_endoHnnA, 0,  sizeof(float) * 2*NENDO*ND3);
	cudaMemset(d_surfPOTi, 0,  sizeof(float) * (NL-2)*2);
	//( cudaMemcpy((d_POTi), (g_POTi), sizeof(float) * NL, cudaMemcpyHostToDevice));
	//( cudaMemcpy((d_der), (g_der), sizeof(float) * NL, cudaMemcpyHostToDevice));
	//( cudaMemcpy((d_endoHnnA), (g_endoHnnA), sizeof(float) * 2*NENDO*ND3, cudaMemcpyHostToDevice));
	//( cudaMemcpy((d_surfPOTi), (g_surfPOTi), sizeof(float) * (NL-2)*2, cudaMemcpyHostToDevice));
}

extern "C" void gpu_BSPitmm_DeviceToHost(float *g_epicPOTold,float *g_POTi,float g_der[NL],float *g_endoHnnA,float *g_surfPOTi)
{
	( cudaMemcpy((g_epicPOTold), (d_epicPOTold), sizeof(float) * Nepic, cudaMemcpyDeviceToHost));
	( cudaMemcpy((g_POTi), (d_POTi), sizeof(float) * NL, cudaMemcpyDeviceToHost));
	( cudaMemcpy((g_der), (d_der), sizeof(float) * NL, cudaMemcpyDeviceToHost));
	( cudaMemcpy((g_endoHnnA),(d_endoHnnA) , sizeof(float) * 2*NENDO*ND3, cudaMemcpyDeviceToHost));
	( cudaMemcpy((g_surfPOTi),(d_surfPOTi) , sizeof(float) * (NL-2)*2, cudaMemcpyDeviceToHost));
}

extern "C" void gpu_dpl_all(short int do_epicPOT,float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL],
	float g_HRTx0,float g_HRTy0,float g_HRTz0,int g_NendoB,int g_NendoC,
	float *g_endoHnnA,short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6],float *g_epicPOTold)
{
	float * d_dpl;
	( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

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

	(cudaFree(d_dpl));


}

extern "C" void gpu_dpl_Nepic(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
	float g_dpl[3],float g_tm[3][6],float *g_epicPOTold)
{
	float * d_dpl;
	( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));
	k_dpl_Nepic<<<Nepic/512+1, 512>>>(d_epicX,d_epicY,d_epicZ,g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,d_dpl,d_epicPOTold,d_tm,Nepic);
	(cudaFree(d_dpl));
}

extern "C" void gpu_dpl_Nendo(float g_posi,float g_posj,float g_posk,float g_HRTx0,float g_HRTy0,float g_HRTz0,
	int g_NendoBC,int g_offset,float g_dpl[3],float *g_endoHnnA,
	short int *g_endoBx,short int *g_endoBy,short int *g_endoBz,float g_tm[3][6])
{
	//k_dpl_Nendo<<<1, g_NendoBC>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,g_offset,g_dpl,d_endoHnnA,d_endoBx,d_endoBy,d_endoBz,d_tm);
	//   numberofb=g_NendoBC;
	//while(g_NendoBC!=0)
	float * d_dpl;
	( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

	if (g_offset<100)
	{
		k_dpl_Nendo<<<6, 512>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,g_offset,d_dpl,d_endoHnnA,d_endoBx,d_endoBy,d_endoBz,d_tm);
	}
	else
	{	k_dpl_Nendo<<<6, 512>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,g_offset,d_dpl,d_endoHnnA,d_endoCx,d_endoCy,d_endoCz,d_tm);
	};

	(cudaFree(d_dpl));
	//k_dpl_Nendo<<<1, (g_NendoBC-512*5)>>>(g_posi,g_posj,g_posk,g_HRTx0,g_HRTy0,g_HRTz0,g_NendoBC,(g_offset+512*5),g_dpl,d_endoHnnA,d_endoBx,d_endoBy,d_endoBz,d_tm);

}
extern "C" void gpu_dpl_nPos_2(float g_posi,float g_posj,float g_posk,float g_dpl[3])
{
	float * d_dpl;
	( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

	k_dpl_nPos_2<<<2, 342>>>(g_posi,g_posj,g_posk,d_dpl,d_r,d_surfPOTi,d_tnd);
	(cudaFree(d_dpl));

}
extern "C" void gpu_dpl_nPos(float g_posi,float g_posj,float g_posk,short int g_nPos,float g_dpl[3],float *g_POTi,float g_der[NL])
{
	float * d_dpl;
	( cudaMalloc((void**) &d_dpl, sizeof(float) * 3));
	( cudaMemcpy(d_dpl, g_dpl, sizeof(float) * 3, cudaMemcpyHostToDevice));

	//float *d_POTi=0, *d_der=0;
	//  ( cudaMalloc((void**) &d_POTi, sizeof(float) * NL));
	//  ( cudaMalloc((void**) &d_der, sizeof(float) * NL));
	//  ( cudaMemcpy((d_POTi), (g_POTi), sizeof(float) * NL, cudaMemcpyHostToDevice));
	//  ( cudaMemcpy((d_der), (g_der), sizeof(float) * NL, cudaMemcpyHostToDevice));

	k_dpl_nPos<<<1, g_nPos>>>(g_posi,g_posj,g_posk,g_nPos,d_dpl,d_POTi,d_der,d_r ,d_rn);

	(cudaFree(d_dpl));

	//k_dpl_nPos<<<1, g_nPos>>>(g_posi,g_posj,g_posk,g_nPos,g_dpl,d_POTi,d_der,d_r,d_rn);

	//( cudaMemcpy((g_POTi), (d_POTi), sizeof(float) * NL, cudaMemcpyDeviceToHost));
	//( cudaMemcpy((g_der), (d_der), sizeof(float) * NL, cudaMemcpyDeviceToHost));		
	//(cudaFree(d_der));
	//(cudaFree(d_POTi));
	//extern "C" void dplpro(float *POTi,const short int NL, const float **r)

	//	float *d_data=0,*d_r[3],;
	//	printf("%f,%f\n", *POTi,*(POTi+1));
	//	for(int i=0;i<3,i++) ( cudaMalloc((void**) &d_data, sizeof(float) * NL*4));
	//	( cudaMalloc((void**) &d_data, sizeof(float) * NL*4));
	//	( cudaMemcpy(d_data,POTi , sizeof(float) * NL*4, cudaMemcpyHostToDevice));
	//		dpl<<<1, 16>>>(d_data);
	//		( cudaMemcpy(POTi, d_data, sizeof(float) * NL*4, cudaMemcpyDeviceToHost));
	//	printf("%f,%f\n", *POTi,*(POTi+1));
	//
	//
}

void XCTcalm(void) {
	//      FILE *fp;
	void wtXCTm(short int,short int,short int,short int);
	void bbDLYm(short int,short int,short int);
	void rdXCTm(short int,short int,short int,short int);

	short int itmp, tmp;
	short int iStm,ires,irp,irel,ist,kBB;
	float phsft,mxDLY,mACCl,icross,delt;
	char mCell,iCell,kCell;
	short int *iACTv[4];
	short int *iACTvOld[4];
	short int *jACTv[4];
	short int *kACTv[4];
	short int *iXCT[NK];
	short int *iXCTapd[NK];
	short int *iXCTOld[NK];
	short int iseqx[12]={-1,-1, 0, 0, 1, 0, 1, 1, 0, 0,-1, 0};
	short int iseqy[12]={ 0, 1, 1, 0, 0, 1, 0,-1,-1, 0, 0,-1};
	short int iseqz[12]={ 0, 0, 0,-1,-1,-1, 0, 0, 0, 1, 1, 1};
	short int ix,iy,iz,jx,jy,jz,iv,l;
	short int jdist,jx0,jy0,jz0,is,ICL,ivel;
	short int iSTOP, iS1S2, dS1S2Old, iCell5Ex;
	long i,j,k,nACTv,mACTv,nACTvOld;
	long nblck,nStep,nbrch;
	// >>>>>>> aniso >>>>>>
	float xani,yani,zani,dani,elp;
	float dxani,dyani,dzani;
	short int itms1=0;
	// ---- for vtr aniso use 
	// storing the ellipsoid propagation times ---
	//--------- maximum excitation time Step: maxXctStep -------------
	for(i=0;i<4;i++) {
		iACTv[i]	= (short int *) malloc(50000*ND3*2);
		iACTvOld[i]	= (short int *) malloc(50000*ND3*2);
		jACTv[i]	= (short int *) malloc(50000*ND3*2);
		kACTv[i]	= (short int *) malloc(50000*ND3*2);
		if((iACTv[i]==NULL)||(iACTvOld[i]==NULL)||
			(jACTv[i]==NULL)||(kACTv[i]==NULL)) {
				MessageBox(NULL,"Out of memory !",NULL,MB_OK);
				return;
		}
	}      
	for(i=0;i<NK;i++) {
		iXCT[i]	= (short int *) malloc(NI*NJ*2);
		iXCTapd[i]	= (short int *) malloc(NI*NJ*2);
		iXCTOld[i] = (short int *) malloc(NI*NJ*2);
		if((iXCT[i]==NULL)||(iXCTOld[i]==NULL)) {
			MessageBox(NULL,"Out of memory !",NULL,MB_OK);
			return;
		}
	}
	for(i=0;i<4;i++) {
		for(j=0;j<50000*ND3;j++) {
			*(iACTv[i]+j)=0;
			*(iACTvOld[i]+j)=0;
			*(jACTv[i]+j)=0;
			*(kACTv[i]+j)=0;
		}
	}    
	// --- file mapXCT is initialized with INFTIME ----
	for(i=0;i<NCYCL;i++) {
		for(j=0;j<50000*ND3;j++) {
			*(mapXCTm[i]+j)=INFTIME;
		}
	}
	for(k=0;k<NK;k++) {
		for(j=0;j<NJ;j++) {
			for(i=0;i<NI;i++) {
				*(iXCT[k]+j*NI+i)=INFTIME;
				*(iXCTapd[k]+j*NI+i)=0;
				*(iXCTOld[k]+j*NI+i)=INFTIME;
			}
		}
	}
	mxcycle=0;
	short int tested[NCELL];

	for(i=0;i<NCELL;i++)
		tested[i]=0;
	for(i=0;i<nttl;i++) {
		jx=ipttl[0][i]; /*<Comment by ALF> pos of ith cell*/
		jy=ipttl[1][i];
		jz=ipttl[2][i];
		iCell=*(mapCell[jz]+jy*NI+jx); /*<Comment by ALF> cell type index */
		if(tested[iCell-1]==0)
		{*(iparm+(iCell-1)*NPARM+18)+=ipttl[3][i];tested[iCell-1]=1;
		if (iCell!=1) {*(iparm+(1-1)*NPARM+18)+=ipttl[3][i];//maxXctStep+=ipttl[3][i];
		}
		}

		//TRACE("\nNTTL (%3d %3d %3d) %2d",jx,jy,jz,iCell);
		// set pacemake time of no. 5 cells  
		if (iCell==5) {
			ipstm[0][i]=100*ND/(ipttl[3][i]+1);
			if((ipstm[0][i]*ipttl[3][i])<100*ND) ipstm[0][i]+=1;
			//ipstm[0][i]=100/(ipttl[3][i]+1);
			//if((ipstm[0][i]*ipttl[3][i])<100) ipstm[0][i]+=1;
			//TRACE("\nCell 5, (%d %d %d) %d %d",jx,jy,jz, ipttl[3][i],ipstm[0][i]);
			continue;
		}
		// iparm(n,18) = BCL  basic cycle length (ms) of pacing
		// iparm(n,20) = inc  increament of BCL(ms/cycle)
		ipstm[0][i]=*(iparm+(iCell-1)*NPARM+17);
		ipstm[1][i]=*(iparm+(iCell-1)*NPARM+19);
		ipstm[2][i]=0;
	}   
	nblck=0;
	ic=0;
	nACTv=0;
	iS1S2=0;
	iCell5Ex=0;

	// ------ stimulus: pacemaker spontanous firing -------	
	while (1) {
		// In this loop, ipttl[3][i] is mainly used	to
		// decide ipstm[0][i] and itself	
		jx=0;
		jy=0;
		jz=0;
		iStm=0;
		excited=0;
		for (i=0;i<nttl;i++) {
			jx=ipttl[0][i];
			jy=ipttl[1][i];
			jz=ipttl[2][i];
			iStm=ipttl[3][i];
			iCell=*(mapCell[jz]+jy*NI+jx);

			//TRACE("\nStimulus (%3d %3d %3d)%2d %d %d",jx,jy,jz,iCell,iStm, mxcycle);
			//TRACE("\nbreak1 mxcycle=%d NCYCL=%d ic=%d iCell=%d, iStm=%d, mS2BN=%d,ipstm=%d",mxcycle, NCYCL,ic, iCell, iStm,*(iparm+(iCell-1)*NPARM+18),ipstm[0][i]);
			if (iCell==5) continue; // ignore BB
			if (iStm != ic) continue;
			// ic: i-th time Step
			// nACTv: number of exitation cells at ic time but cellType != 5 (BB)
			// --- end ---
			//TRACE("\nbreak1 mxcycle=%d NCYCL=%d ic=%d iCell=%d, iStm=%d",mxcycle, NCYCL,ic, iCell, iStm);
			nACTv=nACTv+1;
			*(iACTv[0]+nACTv)=jx;
			*(iACTv[1]+nACTv)=jy;
			*(iACTv[2]+nACTv)=jz;
			*(iACTv[3]+nACTv)=*(iparm+(iCell-1)*NPARM+31);  /*<Comment by ALF> iparm store each cell's parameters*/
			// iparm(n,32): conduction speed 
			wtXCTm(ic,jx,jy,jz);
			//if (jx==101 && jy==77 && jz==6) TRACE("\nA mxcycle=%d at ic=%d, iCell=%d",mxcycle,ic,iCell);
			//if (iCell <3) TRACE("\nA %d %d %d %d %d %d",iCell,jx,jy,jz,ic,nACTv);
			// write to file

			// mxcycle: maximum cycle

			if(mxcycle>=NCYCL) {
				break;
			}
			// --- store current time to iXCT and last time to iXCTOld -->
			*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx); // init is INFTIME
			*(iXCT[jz]+jy*NI+jx)=ic;
			excited=1;

			// Update ipttl[3][i]
			// iparm(n,18) = BCL: basic cycle length (ms) of pacing
			// Normally, only SN has this parameter > 0
			/*if(*(iparm+(iCell-1)*NPARM+17)>0) {
			if ((iS1S2==1) && (mS2BN>1)) {
			itmp=ipttl[3][i]+mS2CL;
			mS2BN--;
			} else {
			itmp=ipttl[3][i]+ipstm[0][i];
			}
			dS1S2Old=ipstm[2][i];
			ipstm[0][i] = ipstm[0][i] + ipstm[1][i];
			iCell5Ex=0;
			// ipstm[1][i] is the step
			// iparm(n,19) = pBN: beat number
			// judge by ipttl[3][i]
			if(itmp>*(iparm+(iCell-1)*NPARM+18)) continue;

			if ((mS2ST/3 > ipttl[3][i]) &&(mS2ST/3 < itmp)) {
			ipttl[3][i]=(short int)(mS2ST/3);
			iS1S2=1;
			} else {
			ipttl[3][i]=itmp;
			}	
			ipstm[2][i]=itmp-ipttl[3][i];
			//TRACE("\nTime=%d, %d, %d, %d, %d %d",ic,itmp,ipttl[3][i],dS1S2Old, ipstm[0][i],ipstm[1][i]);      
			continue;
			}*/

			if(*(iparm+(iCell-1)*NPARM+17)>0) {
				if (iCell==1) {
					itmp=ipttl[3][i]+ipstm[0][i];
					dS1S2Old=ipstm[2][i];
					ipstm[0][i] = ipstm[0][i] + ipstm[1][i];
					iCell5Ex=0;
					if(itmp>*(iparm+(iCell-1)*NPARM+18)) continue;
					ipttl[3][i]=itmp; continue;}
				else
				{

					itmp=ipttl[3][i]+ipstm[0][i];
					dS1S2Old=ipstm[2][i];
					ipstm[0][i] = ipstm[0][i] + ipstm[1][i];
					iCell5Ex=0;
					if(itmp>*(iparm+(iCell-1)*NPARM+18)-ipstm[0][i]+3) continue;
					ipttl[3][i]=itmp;

				}
				continue;
			}

			// iparm(n,24) = ICL: intrinsic cycle length(ms)
			ipttl[3][i] = ipttl[3][i] + *(iparm+(iCell-1)*NPARM+23); 
		}

		// ---- display the excitation number ----		
		// go to next Step
		nblck = nblck + nACTv;
		//TRACE("\nmxcycle =%d Step=%3d, number=%ld nblck=%ld ",mxcycle,ic,nACTv, nblck);
		ic = ic + 1;
		//TRACE("\nbreak2 ic=%d maxXctStep=%d ",ic, maxXctStep);
		if (ic>=maxXctStep) break;
		if (nACTv == 0) continue;

		/**
		* very important
		*/
		// --------- propagation (2000)------------>
		nACTvOld=0;
		// nACTv: at moment t, the number of excited cells 
		for (i=1;i<=nACTv;i++) {
			excited=1;
			ix=*(iACTv[0]+i);
			iy=*(iACTv[1]+i);
			iz=*(iACTv[2]+i);
			iv=*(iACTv[3]+i);
			iCell=*(mapCell[iz]+iy*NI+ix);
			//if (ix == 64 && iy == 50 && iz == 64) TRACE("\nB AVN %d",iCell);

			//----------- low conduction speed part ----------->
			// iparm(n,32): conduction speed 
			if (*(iparm+(iCell-1)*NPARM+31)<=0) continue; 
			if (iCell==5) iCell5Ex=1;
			//if (iCell==8) TRACE("\nCell=8  %d, %d, %d, ic=%d, %d %d",ix,iy,iz,ic,iv,mBCL);
			// 100 = Conduction Speed of ATR?
			if (iv<100) {
				nACTvOld=nACTvOld+1;
				*(iACTvOld[0]+nACTvOld)=ix;
				*(iACTvOld[1]+nACTvOld)=iy;
				*(iACTvOld[2]+nACTvOld)=iz;
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)+*(mapSpeed[iz]+iy*NI+ix); //added by zhu
				if (iCell==5) {
					/*ibbDLY=0;
					bbDLYm(ix,iy,iz);
					if (ibbDLY>0)
					*(iACTvOld[3]+nACTvOld)=iv+ibbDLY;
					TRACE("\nBB, %d",*(iACTvOld[3]+nACTvOld));            
					*/

					ibbDLY=0;
					// Add for BB interval by hui wang
					ibbSTEP=0;
					bbDLYm(ix,iy,iz);

					// End of add for BB interval by hui wang, modified by zhu


					if (ibbDLY>0) {ibbSTEP+=nbbSTEP;ibbDLY=100*ND/(ibbSTEP+1);}

					if(ibbDLY>0 && (ibbDLY*ibbSTEP)<100*ND) ibbDLY+=1;

					if (ibbDLY>0)
						*(iACTvOld[3]+nACTvOld)=iv+ibbDLY;
					else
						*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31);

					continue;
				}
				/*if (iCell==3 || iCell==6) {
				if (*(iXCTOld[iz]+iy*NI+ix)==INFTIME)
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31);
				else {

				irel = *(iXCT[iz]+iy*NI+ix)-*(iXCTOld[iz]+iy*NI+ix)-(*(iparm+NPARM*(iCell-1)+4)+*(mapAPD[iz]+iy*NI+ix))/3;

				//irel = *(iXCT[iz]+iy*NI+ix)-*(iXCTOld[iz]+iy*NI+ix)-(*(iparm+NPARM*(iCell-1)+4))/3;
				irel = 3*irel;

				if (irel<*(iparm+NPARM*(iCell-1)+5)) {
				tmp=100+*(iparm+NPARM*(iCell-1)+32)
				-irel*(*(iparm+NPARM*(iCell-1)+32))/(*(iparm+NPARM*(iCell-1)+5));
				if (tmp!=0) {
				ivel = 100*(*(iparm+NPARM*(iCell-1)+31))/tmp;
				} else {
				ivel=*(iparm+NPARM*(iCell-1)+31);
				}
				} else {  
				//  <--- time of RRP stored in iparm(6) ---
				ivel=*(iparm+NPARM*(iCell-1)+31);
				}

				*(iACTvOld[3]+nACTvOld)=iv+ivel;}

				}*/

				/*else if (iCell==3) {
				if (iCell5Ex==0) {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)- dS1S2Old/20;
				TRACE("\nCell=3 E dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else {
				if (mBCL<600&&dS1S2Old<140/3) {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)- (dS1S2Old+67)/33;						
				TRACE("\nCell=3 A dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else if (mBCL<600&&dS1S2Old>=140/3) {
				*(iACTvOld[3]+nACTvOld)=iv;
				TRACE("\nCell=3 B dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else if (mBCL>=600&&dS1S2Old<=210/3) {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31);
				TRACE("\nCell=3 C dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} else {
				*(iACTvOld[3]+nACTvOld)=iv+*(iparm+NPARM*(iCell-1)+31)- dS1S2Old/12;
				TRACE("\nCell=3 D dS1S2Old %d, %d, ic=%d, %d %d",dS1S2Old,*(iACTvOld[3]+nACTvOld),ic,iv,mBCL);
				} 						
				}
				}*/
				continue; 
			}

			// ------- neighbourhood search (2100) -------->
			// go to iv > 100 situation and set ires = the part of iv < 100
			ires=iv-100*(int)(iv/100);

			for (l=0;l<12;l++) {
				jx=ix+iseqx[l];
				if ((jx<=-1)||(jx>(NI-1))) continue; // >=0 <NI
				jy=iy+iseqy[l];
				if ((jy<=-1)||(jy>(NJ-1))) continue; // >=0 <NJ
				jz=iz+iseqz[l];
				if ((jz<=-1)||(jz>(NK-1))) continue; // >=0 <NK
				// >>>>> aniso: within the ellpisoid ? >>>>>>>>>>>>
				if (ANISO==1 && iCell==7) {
					dani=local(ix,iy,iz);
					//TRACE("\nx,y,z,dani, %2d %2d %2d %f",ix,iy,iz,dani);
					// -- if can't solve local coordinates, treat as isotropic -->
					if (dani > 0.0001) {
						//lctran(iseqx[l],iseqy[l],iseqz[l],dani,xani,yani,zani);
						xani=iseqx[l]*tmswf[0][0]+iseqy[l]*tmswf[0][1]+iseqz[l]*tmswf[0][2];
						yani=iseqx[l]*tmswf[1][0]+iseqy[l]*tmswf[1][1]+iseqz[l]*tmswf[1][2];
						zani=iseqx[l]*tmswf[2][0]+iseqy[l]*tmswf[2][1]+iseqz[l]*tmswf[2][2];
						dxani=xani*yaxis[1]*zaxis[2]+yani*yaxis[2]*zaxis[0]
						+zani*yaxis[0]*zaxis[1]-zani*yaxis[1]*zaxis[0]
						-xani*yaxis[2]*zaxis[1]-yani*yaxis[0]*zaxis[2];

						dyani=xaxis[0]*yani*zaxis[2]+xaxis[1]*zani*zaxis[0]
						+xaxis[2]*xani*zaxis[1]-xaxis[2]*yani*zaxis[0]
						-xaxis[0]*zani*zaxis[1]-xaxis[1]*xani*zaxis[2];

						dzani=xaxis[0]*yaxis[1]*zani+xaxis[1]*yaxis[2]*xani
							+xaxis[2]*yaxis[0]*yani-xaxis[2]*yaxis[1]*xani
							-xaxis[0]*yaxis[2]*yani-xaxis[1]*yaxis[0]*zani;
						xani=dxani/dani;
						yani=dyani/dani;
						zani=dzani/dani;	
						// itms=maps(ix,iy,iz)+1
						//TRACE("\nd  %f %f %f %f",dxani,xani,yaxis[0],zaxis[0]);
						itms1=*(iXCTapd[iz]+iy*NI+ix);    								
						elp=xani*xani/vt2[itms1]+
							yani*yani/vt2[itms1]+
							zani*zani/vl2[itms1];
						// write(0,*) x,y,z,elp
						// TRACE("\n %d %f",itms1,elp);
						if (elp > 1.0) continue;
					}
				}
				// <<<<<<<<<<<<<<<<<<<< aniso <<<<<<<<<<<<<<<<<<<
				mCell=*(mapCell[jz]+jy*NI+jx);
				if ((iCell<=7)&&(mCell<=7)&&(((iCell-mCell)>1)||
					((iCell-mCell)<-1))) continue;
				if ((*(iparm+NPARM*(mCell-1)+33)>0)&&( (mCell>7 && iCell>7 && mCell!=iCell) || (mCell<=7 && iCell>mCell) || (mCell>7 && !(iCell==mCell || iCell==2)))) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)>0)&&( (mCell<=7 && iCell>mCell) || (mCell>7 && !(iCell==mCell || iCell==2)))) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)>0)&&(iCell>mCell)) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)<0)&&(iCell<mCell)) continue;
				//if ((*(iparm+NPARM*(mCell-1)+33)<0)&&( (mCell<=7 && iCell<mCell) || (mCell>7 && !(iCell==mCell || iCell==7)))) continue;
				if ((*(iparm+NPARM*(mCell-1)+33)<0)&&( (mCell>7 && iCell>7 && mCell!=iCell) || (mCell<=7 && iCell<mCell) || (mCell>7 && !(iCell==mCell || iCell==7)))) continue;

				//if (jx == 64 && jy == 50 && jz == 64) TRACE("\nC AVN %d",mCell);
				if (mCell<=0) continue; // continue;
				if (mCell>=15) continue; // continue;
				if (mCell==5) iCell5Ex=1;
				//if (iCell==8) TRACE("\nCell=8  %d, %d, %d, ic=%d, %d %d",jx,jy,jz,ic);;
				// --- coupling interval ------>
				idltt=ic-*(iXCT[jz]+jy*NI+jx);
				if (*(iXCT[jz]+jy*NI+jx)==INFTIME) idltt=INFTIME;

				// --- change in cycle length ------>
				idltc=idltt+*(iXCTOld[jz]+jy*NI+jx)-*(iXCT[jz]+jy*NI+jx);
				if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME) idltc=0;
				if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
					*(iparm+(1-1)*NPARM+23)>0)
					idltc = ic-*(iXCT[jz]+jy*NI+jx)-*(iparm+(1-1)*NPARM+23);


				//      rdXCT(ic,jx,jy,jz);
				// irp = time in phase 2 + mapACT/3 +plateau of potential in phase 3 *idltc/100
				// --- absolute refractory period ------>
				//irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3
				//	+*(iparm+(mCell-1)*NPARM+10)*idltc/100;
				irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3;


				//if (mCell==6)
				//        irp=(*(iparm+NPARM*(mCell-1)+4))/3;
				irel=idltt-irp;


				// ++++++++ in absolute refractory period ? +++++++
				if (irel<=0) continue;
				//if (*(mapAPD[jz]+jy*NI+jx)>20) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>20 && idltc<0) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>30 && idltc<0) idltc = 3*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>30 && idltc>0) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>40 && idltc>0) idltc = 2*idltc;
				//if (*(mapAPD[jz]+jy*NI+jx)>40 && idltc>0) idltc = 2*idltc;
				//if (mCell==3) TRACE("\nCell=3  %d, %d, %d,%d,%d,%d",irp,ic,*(iXCT[jz]+jy*NI+jx),*(mapAPD[jz]+jy*NI+jx),*(iparm+(mCell-1)*NPARM+10),idltc);


				*(mapAPD[jz]+jy*NI+jx) += *(iparm+(mCell-1)*NPARM+10)*idltc*3/100; //added by Zhu

				//if (mCell==3) TRACE("\nCell=3  %d, %d, %d",*(mapAPD[jz]+jy*NI+jx)/3,idltc,irp);

				if (*(iXCT[jz]+jy*NI+jx)==INFTIME && mCell==3) {irel=INFTIME;*(mapAPD[jz]+jy*NI+jx)=0;}





				// --- find automaticity in stimul data ----
				// iparm(n,24), ICL: intrinsic cycle length (ms)
				iSTOP =0;
				if (*(iparm+NPARM*(mCell-1)+23)>0) {  // !=0 August 10, 1996
					// <--- next touch time should be beyound ARP of the cell --

					for (is=0;is<nttl;is++) {
						if (jx!=ipttl[0][is]) continue; 
						if (jy!=ipttl[1][is]) continue; 
						if (jz!=ipttl[2][is]) continue; 
						// --- iparm(23) used for adjusting intermediate change
						// of EP intrinsic cycle length --->
						ICL = *(iparm+NPARM*(mCell-1)+23);
						ist = ic-*(iXCT[jz]+jy*NI+jx);

						// PRT: protection indicator
						// --- no protection ---->
						if (*(iparm+NPARM*(mCell-1)+24)==0) {
							if (ist<=irp) continue; //{iSTOP=1;break;}
							//if (iSTOP==1) 
							ipttl[3][is]=ic+ICL; // ICL/3
							/******************/
							nACTvOld=nACTvOld+1; 
							*(iACTvOld[0]+nACTvOld)=jx;
							*(iACTvOld[1]+nACTvOld)=jy;
							*(iACTvOld[2]+nACTvOld)=jz;
							*(iACTvOld[3]+nACTvOld)=*(iparm+NPARM*(mCell-1)+31)+ires;
							wtXCTm(ic,jx,jy,jz);
							if (mxcycle>=NCYCL) {iSTOP=1;break;}
							//if (ic==*(iXCT[jz]+jy*NI+jx)) continue;
							*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
							*(iXCT[jz]+jy*NI+jx)=ic;//added to by zhu
							irel=0;
							excited=1;
							TRACE("\n %d, %d %d %d %d %d",*(iXCTOld[jz]+jy*NI+jx),*(iXCT[jz]+jy*NI+jx),ic,jx,jy,jz);
							/******************/

							//iSTOP=1;
							continue; //break;  // rewrite condition
						}
						if (idltt==INFTIME) continue;

						//ist = ic-*(iXCT[jz]+jy*NI+jx);
						//            if (ist<=irp)        goto loop21; // August 10, 1996
						if (ist<=irp) continue; //{iSTOP=1;break;}
						phsft =(float)100.*(idltt/ICL);
						mxDLY =(float)*(iparm+NPARM*(mCell-1)+25);
						mACCl =(float)*(iparm+NPARM*(mCell-1)+26);
						if (mxDLY == 0 && mACCl == 0) continue;
						icross=(float)*(iparm+NPARM*(mCell-1)+27);
						if (icross == 0 || icross == 100) continue;
						if (phsft<=icross) 
							delt=phsft*mxDLY/icross;
						else 
							delt=mACCl-(phsft-icross)*mACCl/(100-icross);
						// -- store touch time --->
						// -- modify next stimulating time --->
						ipttl[3][is]=ipttl[3][is]+(int)(ICL*delt/100);

						//TRACE("\ntime=%4d,ixt=%4d,idltt=%4d,icl=%4d,phsft=%4d,intermediate=%4d",
						//	ic, *(iXCT[jz]+jy*NI+jx),idltt,ICL, (int)phsft, ipttl[3][is]); 
						// change value after each touch time 
						// avoiding from successive modification by surrounding cells
						if (ic==*(iXCT[jz]+jy*NI+jx)) continue;
						*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
						*(iXCT[jz]+jy*NI+jx)=ic;
						irel=0;
						excited=1;
						//iSTOP=1;
						continue; //break;  // rewrite condition
					}
				} 
				if (iSTOP==1) continue;
				if (irel==0) continue;


				// +++++ special processing for BB +++++
				if (mCell==5) {


					*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
					*(iXCT[jz]+jy*NI+jx)=ic;

					// Add for BB interval by hui wang modified by Zhu
					// variable ibbSTEP, nbbSTEP are added to store steps by first BB
					// ibbSTEP is a function in bbDLYm(i,j,k)
					nbbSTEP=0;
					ibbDLY=0;
					ibbSTEP=0;
					bbDLYm(jx,jy,jz);
					nbbSTEP=ibbSTEP;
					// end of add for BB interval by hui wang
					//ic+=10; // add by hw, BB interval
					//TRACE("\n nHB = %d, ic= %d",nHB,ic);
					for(kBB=0;kBB<nBB;kBB++) { 
						jx=iBB[0][kBB];
						jy=iBB[1][kBB];
						jz=iBB[2][kBB];
						nACTvOld=nACTvOld+1;
						*(iACTvOld[0]+nACTvOld)=jx;
						*(iACTvOld[1]+nACTvOld)=jy;
						*(iACTvOld[2]+nACTvOld)=jz;
						//*(iACTvOld[3]+nACTvOld)=100;
						ibbDLY=0;
						// Add for BB interval by hui wang,modified by zhu
						ibbSTEP=0;
						bbDLYm(jx,jy,jz);
						ibbSTEP+=nbbSTEP;
						ibbDLY=100*ND/(ibbSTEP+1);
						if((ibbDLY*ibbSTEP)<100*ND) ibbDLY+=1;
						// End of add for BB interval by hui wang
						*(iACTvOld[3]+nACTvOld)=ibbDLY;

						wtXCTm(ic,jx,jy,jz);
						//if (jx==101 && jy==77 && jz==6) TRACE("\nB mxcycle=%d at ic=%d, iCell=%d",mxcycle,ic,mCell);
						//if (mCell >2 && mCell <6) TRACE("\nB %d %d %d %d %d %d",mCell,jx,jy,jz,ic,nACTvOld);
						if (mxcycle>=NCYCL) break;
						*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
						*(iXCT[jz]+jy*NI+jx)=ic;
						excited=1;
					}
					continue;
				} 


				nACTvOld=nACTvOld+1; 
				*(iACTvOld[0]+nACTvOld)=jx;
				*(iACTvOld[1]+nACTvOld)=jy;
				*(iACTvOld[2]+nACTvOld)=jz;
				wtXCTm(ic,jx,jy,jz);
				//TRACE("\nbreak3 mxcycle=%d NCYCL=%d ",mxcycle, NCYCL);
				if (mxcycle>=NCYCL) break;
				*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
				*(iXCT[jz]+jy*NI+jx)=ic;

				irel = 3*irel;
				//if (*(iXCTOld[jz]+jy*NI+jx)!=INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
				//			irel>*(iparm+NPARM*(mCell-1)+5))
				//	*(mapAPD[jz]+jy*NI+jx)=0;

				// time of RRP stored in iparm(6)
				if ((irel)<*(iparm+NPARM*(mCell-1)+5)) {
					tmp=100+*(iparm+NPARM*(mCell-1)+32)
						-irel*(*(iparm+NPARM*(mCell-1)+32))/(*(iparm+NPARM*(mCell-1)+5));
					if (tmp!=0) {
						ivel = 100*(*(iparm+NPARM*(mCell-1)+31))/tmp;
					} else {
						ivel=*(iparm+NPARM*(mCell-1)+31);
					}
				} else {  
					//  <--- time of RRP stored in iparm(6) ---
					ivel=*(iparm+NPARM*(mCell-1)+31);
				}
				*(mapSpeed[jz]+jy*NI+jx)=ivel-*(iparm+NPARM*(mCell-1)+31);//added by Zhu

				// test results
				//TRACE("\nmcell=%4d,ic=%4d,idltt=%4d,idltc=%4d,ivel=%4d",mCell,ic,idltt,idltc,ivel);

				if (iCell!=mCell) {
					if (mCell == 5) {
						bbDLYm(jx,jy,jz);
						*(iACTvOld[3]+nACTvOld)=ibbDLY;
						//TRACE("\n BB2=%d, %d %d (%d %d %d) ic=%d ",*(iACTvOld[3]+nACTvOld),iv,ibbDLY,ix,iy,iz, ic);
						continue;
					}
					*(iACTvOld[3]+nACTvOld)=ivel;
					continue;
				}
				*(iACTvOld[3]+nACTvOld)=ivel+ires;				
			}
			// <------- END of neighbourhood search (2100) -----
			// >>>>>>>> anisotropy >>>>>
			if (ANISO==1 && iCell == 7) {
				// ltrat==2;
				if (*(iXCTapd[iz]+iy*NI+ix) < 2) {
					nACTvOld=nACTvOld+1;
					*(iACTvOld[0]+nACTvOld)=ix;
					*(iACTvOld[1]+nACTvOld)=iy;
					*(iACTvOld[2]+nACTvOld)=iz;
					//*(iACTvOld[3]+nACTvOld)=ires+ 
					//	*(iparm+NPARM*(iCell-1)+31);
					*(iACTvOld[3]+nACTvOld)=ires+ 
						*(iparm+NPARM*(iCell-1)+31)+*(mapSpeed[iz]+iy*NI+ix);
					*(iXCTapd[iz]+iy*NI+ix)+=1;
				} else {
					*(iXCTapd[iz]+iy*NI+ix)=0;
				}
			}
			// <<<<<<<<<<<
		}
		// <------- END of propagation (2000) -----

		//  +++++++++++ for high speed ++++++++
		mACTv=nACTvOld;
		// ------- propagation (1000) -------->
		for(i=1;i<=nACTvOld;i++) {
			idist=(int)(*(iACTvOld[3]+i)/100);       
			if (idist<2) continue; 
			*(jACTv[0]+1)=*(iACTvOld[0]+i);
			*(jACTv[1]+1)=*(iACTvOld[1]+i);
			*(jACTv[2]+1)=*(iACTvOld[2]+i);
			ires=*(iACTvOld[3]+i)-idist*100;
			nStep=0;
			nbrch=1;
			jdist=1;
			while (1) {   
				for (j=1;j<=nbrch;j++) {
					jx0=*(jACTv[0]+j);
					jy0=*(jACTv[1]+j);
					jz0=*(jACTv[2]+j);
					mCell=*(mapCell[jz0]+jy0*NI+jx0);
					if (mCell==5) iCell5Ex=1;

					for (l=0;l<12;l++) {
						jx=jx0+iseqx[l];
						if ((jx<=-1)||(jx>(NI-1))) continue;   // <0 or >=NI
						jy=jy0+iseqy[l];
						if ((jy<=-1)||(jy>(NJ-1))) continue;   // <0 or >=NJ
						jz=jz0+iseqz[l];
						if ((jz<=-1)||(jz>(NK-1))) continue;   // <0 or >=NK
						kCell = *(mapCell[jz]+jy*NI+jx);
						//if (jx == 64 && jy == 50 && jz == 64) TRACE("\nE AVN %d",kCell);

						if (kCell != mCell) continue; 
						//++++++++ in effective refractory period ? +++++++

						// >>>>> aniso: within the ellpisoid ? >>>>>>>>>>>>
						if (ANISO==1 && mCell==7) {
							dani=local(jx0,jy0,jz0);
							//TRACE("\nx,y,z,dani, %2d %2d %2d %f",ix,iy,iz,dani);
							// -- if can't solve local coordinates, treat as isotropic -->
							if (dani > 0.0001) {
								//lctran(iseqx[l],iseqy[l],iseqz[l],dani,xani,yani,zani);
								xani=iseqx[l]*tmswf[0][0]+iseqy[l]*tmswf[0][1]+iseqz[l]*tmswf[0][2];
								yani=iseqx[l]*tmswf[1][0]+iseqy[l]*tmswf[1][1]+iseqz[l]*tmswf[1][2];
								zani=iseqx[l]*tmswf[2][0]+iseqy[l]*tmswf[2][1]+iseqz[l]*tmswf[2][2];
								dxani=xani*yaxis[1]*zaxis[2]+yani*yaxis[2]*zaxis[0]
								+zani*yaxis[0]*zaxis[1]-zani*yaxis[1]*zaxis[0]
								-xani*yaxis[2]*zaxis[1]-yani*yaxis[0]*zaxis[2];

								dyani=xaxis[0]*yani*zaxis[2]+xaxis[1]*zani*zaxis[0]
								+xaxis[2]*xani*zaxis[1]-xaxis[2]*yani*zaxis[0]
								-xaxis[0]*zani*zaxis[1]-xaxis[1]*xani*zaxis[2];

								dzani=xaxis[0]*yaxis[1]*zani+xaxis[1]*yaxis[2]*xani
									+xaxis[2]*yaxis[0]*yani-xaxis[2]*yaxis[1]*xani
									-xaxis[0]*yaxis[2]*yani-xaxis[1]*yaxis[0]*zani;
								xani=dxani/dani;
								yani=dyani/dani;
								zani=dzani/dani;	
								// itms=maps(ix,iy,iz)+1
								//TRACE("\nd  %f %f %f %f",dxani,xani,yaxis[0],zaxis[0]);
								itms1=*(iXCTapd[jz0]+jy0*NI+jx0);    								
								elp=xani*xani/vt2[itms1]+
									yani*yani/vt2[itms1]+
									zani*zani/vl2[itms1];
								// write(0,*) x,y,z,elp
								// TRACE("\n %d %f",itms1,elp);
								if (elp > 1.0) continue;
							}
						}

						idltt=ic-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME) idltt=INFTIME;

						// --- change in cycle length ------>
						idltc=idltt+*(iXCTOld[jz]+jy*NI+jx)-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME) idltc=0;
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
							*(iparm+(1-1)*NPARM+23)>0)
							idltc = ic-*(iXCT[jz]+jy*NI+jx)-*(iparm+(1-1)*NPARM+23);

						//      rdXCT(ic,jx,jy,jz);
						// irp = time in phase 2 + mapACT/3 +plateau of potential in phase 3 *idltc/100
						// --- absolute refractory period ------>
						//irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3
						//	+*(iparm+(mCell-1)*NPARM+10)*idltc/100;
						irp=(*(iparm+NPARM*(mCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3;


						//if (mCell==6)
						//        irp=(*(iparm+NPARM*(mCell-1)+4))/3;
						irel=idltt-irp;
						if (irel<=0) continue; // continue;
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME && mCell==3) {irel=INFTIME;*(mapAPD[jz]+jy*NI+jx)=0;}
						*(mapAPD[jz]+jy*NI+jx) += *(iparm+(mCell-1)*NPARM+10)*idltc*3/100; //added by Zhu
						/*
						idltt=ic-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME)	idltt=INFTIME;
						idltc=idltt+*(iXCTOld[jz]+jy*NI+jx)-*(iXCT[jz]+jy*NI+jx);
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME) idltc=0;
						if (*(iXCTOld[jz]+jy*NI+jx)==INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
						*(iparm+(1-1)*NPARM+23)>0)
						idltc = ic-*(iXCT[jz]+jy*NI+jx)-*(iparm+(1-1)*NPARM+23);
						//      rdXCT(ic,jx,jy,jz);
						// irp=(*(iparm+NPARM*(kCell-1)+4)+*(mapACT[jz]+jy*NI+jx))/3+
						// 		*(iparm+(kCell-1)*NPARM+10)*idltc/100;
						//irp=(*(iparm+NPARM*(kCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3+
						//	*(iparm+(kCell-1)*NPARM+10)*idltc/100;
						irp=(*(iparm+NPARM*(kCell-1)+4)+*(mapAPD[jz]+jy*NI+jx))/3;
						*(mapAPD[jz]+jy*NI+jx) += *(iparm+(kCell-1)*NPARM+10)*idltc*3/100; //added by Zhu

						irel=idltt-irp;
						if (*(iXCT[jz]+jy*NI+jx)==INFTIME) irel=INFTIME;
						*/

						irel = 3*irel;
						//if (*(iXCTOld[jz]+jy*NI+jx)!=INFTIME && *(iXCT[jz]+jy*NI+jx)!=INFTIME && 
						//	irel>=*(iparm+NPARM*(kCell-1)+5))
						//    *(mapAPD[jz]+jy*NI+jx)=0;

						if ((irel)<*(iparm+NPARM*(kCell-1)+5)) {
							tmp=100+*(iparm+NPARM*(mCell-1)+32)
								-irel*(*(iparm+NPARM*(mCell-1)+32))/(*(iparm+NPARM*(mCell-1)+5));
							if (tmp!=0) {
								ivel = 100*(*(iparm+NPARM*(mCell-1)+31))/tmp;
							} else {
								ivel=*(iparm+NPARM*(mCell-1)+31);
							}
						} else {
							ivel=*(iparm+NPARM*(kCell-1)+31);
						}

						*(mapSpeed[jz]+jy*NI+jx)=ivel-*(iparm+NPARM*(kCell-1)+31);//added by Zhu


						nStep=nStep+1;
						*(kACTv[0]+nStep)=jx;
						*(kACTv[1]+nStep)=jy;
						*(kACTv[2]+nStep)=jz;
						// nStep++;
						mACTv=mACTv+1;
						*(iACTvOld[0]+mACTv)=jx;
						*(iACTvOld[1]+mACTv)=jy;
						*(iACTvOld[2]+mACTv)=jz;
						*(iACTvOld[3]+mACTv)=ivel+ires;

						// mACTv++;
						// TRACE(" D%d,",mACTv);
						wtXCTm(ic,jx,jy,jz);
						//if (jx==101 && jy==77 && jz==6) TRACE("\nD mxcycle=%d at ic=%d, iCell=%d",mxcycle,ic,kCell);
						//if (kCell >2 && kCell <6) TRACE("\nD %d %d %d %d %d %d",kCell,jx,jy,jz,ic,mACTv);
						//TRACE("\nbreak4 mxcycle=%d NCYCL=%d ",mxcycle, NCYCL);
						if (mxcycle>=NCYCL) {
							//TRACE("\nbreak5 iSTOP=%d mxcycle=%d,NCYCL=%d",iSTOP, mxcycle, NCYCL);
							iSTOP =1;
							break;
						}
						*(iXCTOld[jz]+jy*NI+jx)=*(iXCT[jz]+jy*NI+jx);
						*(iXCT[jz]+jy*NI+jx)=ic;
						excited=1;


					}

					/*// >>>>>>>> anisotropy >>>>>
					if (ANISO==1 && mCell == 7) {
					// ltrat==3;
					if (*(iXCTapd[jz0]+jy0*NI+jx0) < 3) {
					mACTv=mACTv+1;
					*(iACTvOld[0]+mACTv)=jx0;
					*(iACTvOld[1]+mACTv)=jy0;
					*(iACTvOld[2]+mACTv)=jz0;
					*(iACTvOld[3]+mACTv)=ivel+ires;
					//*(iACTvOld[3]+nACTvOld)=ires+ 
					//	*(iparm+NPARM*(iCell-1)+31);
					*(iACTvOld[3]+nACTvOld)=ires+ 
					*(iparm+NPARM*(mCell-1)+31)+*(mapSpeed[jz0]+jy0*NI+jx0);
					*(iXCTapd[jz0]+jy0*NI+jx0)+=1;
					} else {
					*(iXCTapd[jz0]+jy0*NI+jx0)=0;
					}
					}
					// <<<<<<<<<<<*/
					if (iSTOP ==1) break;
				}
				if (iSTOP ==1) break;

				if (nStep==0) break; // continue;
				jdist=jdist+1;
				if (jdist>=idist) break; // continue;
				for(k=1;k<=nStep;k++) {
					*(jACTv[0]+k)=*(kACTv[0]+k);
					*(jACTv[1]+k)=*(kACTv[1]+k);
					*(jACTv[2]+k)=*(kACTv[2]+k);
				}
				nbrch=nStep;
				nStep=0;
			}
		}
		//TRACE("\nbreak5 iSTOP=%d ",iSTOP);
		if (iSTOP ==1) break;
		// <------- END of propagation (1000) -------------
		if (excited == 0) break;
		nACTv=mACTv;
		// nblck=nblck+nACTv;

		for(i=1;i<=nACTv;i++) {
			for(j=0;j<4;j++) {
				*(iACTv[j]+i)=*(iACTvOld[j]+i);
			}
		}

	} // END of whole while loop 
	TRACE("\nmxcycle=%d",mxcycle);

	mxcycle++; // hui


	// add HB info
	for (itmp=0; itmp<50*ND; itmp++) {
		for (tmp=0;tmp<NCYCL;tmp++) { 
			vHB[tmp][itmp]=0;
		}
	}

	for (itmp=0; itmp<nHB; itmp++) {
		l=iHB[0][itmp];
		j=iHB[1][itmp];
		k=iHB[2][itmp];
		if (itmp==0) i=*(locXCT[k]+j*NJ+l); // Consider only the point near AV Node
		for (tmp=0;tmp<mxcycle;tmp++) { 
			vHB[tmp][itmp]=*(mapXCTm[tmp]+i);
		}
	}

	// Save 
	CFile f;
	CFileException e;
	//short int index;
	//index=filepath.FindOneOf(".");
	//filepath.SetAt(index+1,'x');
	//filepath.SetAt(index+2,'c');
	//filepath.SetAt(index+3,'t');

	if (!f.Open( dataPath+"tour.xct ", CFile::modeCreate | CFile::modeWrite, &e )) {
#ifdef _DEBUG
		afxDump << "File could not be opened " << e.m_cause << "\n";
#endif
	}
	//f.Write(&mxcycle,2);	
	f.Write(&miBN,2);	
	f.Write(&ic,2);
	f.Write(&totalCell,4);

	for(j=0;j<mxcycle;j++) {    
		for(i=0;i<totalCell;i++) f.Write(mapXCTm[j]+i,2);
	}    
	f.Close();

	/*	
	FILE * iow;
	iow=fopen("fpMapXCTm.txt","wt");	
	if (iow == NULL) {		
	fprintf(stderr, "Open .txt for write failed! \n");		
	return;		
	}	

	long temploc;

	temploc=*(locXCT[45]+22*NJ+33);
	fprintf(iow,"33 22 45 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[40]+30*NJ+32);
	fprintf(iow,"32 30 40 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[48]+20*NJ+30);
	fprintf(iow,"30 20 48 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[56]+8*NJ+26);
	fprintf(iow,"26  8 56 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[62]+10*NJ+21);
	fprintf(iow,"21 10 62 %3d\n",*(mapXCTm[0]+temploc));
	temploc=*(locXCT[62]+30*NJ+13);
	fprintf(iow,"13 30 62 %3d\n",*(mapXCTm[0]+temploc));

	for(l=0;l<mxcycle;l++) {    
	fprintf(iow,"l=%d\n",l);
	for(k=0;k<NK;k++) {
	for(j=0;j<NJ;j++) {
	for(i=NI-1;i>-1;i--) {
	temploc = *(locXCT[k]+j*NJ+i);
	if (temploc < 0) fprintf(iow,"    ");
	else fprintf(iow,"%3d ",*(mapXCTm[l]+temploc));
	}
	fprintf(iow,"j=%d\n",j);
	}
	fprintf(iow,"k=%d\n",k);
	}
	}
	fclose(iow);
	*/

	for(i=0;i<4;i++) {
		free(iACTv[i]);
		free(iACTvOld[i]);
		free(jACTv[i]);
		free(kACTv[i]);
	}       
	for(i=0;i<NK;i++) {
		free(iXCT[i]);
		free(iXCTapd[i]);
		free(iXCTOld[i]);
	}
}