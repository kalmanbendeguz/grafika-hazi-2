//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kalman Bendeguz Bence
// Neptun : PTW6BD
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

/* Minden programreszt ahol nincs forrasmegjeloles, en irtam. A program teljesitmenyet befolyasolni 
vagy megfigyelni leggyorsabban az alabbi konstansok valtoztatasaval lehet: */

const bool dbrender = 0; /* Ha ez nem 0, akkor minden pixelsor utan kiirja annak a sornak a szamat (tehat osszesen 600-szor), debug celbol. */
const float epsilon = 0.0001f; /* A numerikus pontossagot adja meg, minel kisebb, annal gyorsabb a program es reszletesebb a kep,
							   de ha tul kicsire allitjuk, szemcses lesz a kep*/
const int nv = 14; /* A mintapontok szama */
const int depthOfRecursion = 3; /* A rekurzio melysege */

float rnd() { return (float)rand() / RAND_MAX; }

const float r_velux = 1.0;
const vec3 normalOfDisk(0, 0, -1.0);
const vec3 one(1, 1, 1);

/* Kiadott videobol */
enum MaterialType { ROUGH, REFLECTIVE };

/* Kiadott videobol */
struct Material {
	vec3 ka, kd, ks;
	float shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) {
		type = t;
	}
};

/* Kiadott videobol */
struct RoughMaterial : Material {
	RoughMaterial(const vec3& _kd, const vec3& _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(const vec3& num, const vec3& denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

/* Kiadott videobol */
struct ReflectiveMaterial : Material {
	ReflectiveMaterial(const vec3& n, const vec3& kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
	}
};

/* raytrace.cpp - bol */
struct Hit { 
	float t;
	vec3 position;
	vec3 normal;
	Material* material;
	Hit() { t = -1; }
};

/* raytrace.cpp - bol */
struct Ray {
	vec3 start, dir;
	Ray(const vec3& _start,const vec3& _dir) {
		start = _start; dir = normalize(_dir);
	}
};

/* raytrace.cpp - bol */
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Quadrics : public Intersectable {
	vec3 center;
	mat4 Q;

	Quadrics() {};

	Quadrics(const vec3& _center, const mat4& _Q, Material* _material) {
		center = _center;
		Q = _Q;
		material = _material;
	}

	/* Eloadasdiarol */
	float f(const vec4& r) {
		return dot(r * Q, r);
	}
	 
	/* Eloadasdiarol */
	vec3 gradf(const vec4& r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 S(ray.start.x - center.x, ray.start.y - center.y, ray.start.z - center.z, 1);

		float a = f(D);
		float b = dot(S * Q, D) + dot(D * Q, S);
		float c = f(S);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec4 cross(hit.position.x, hit.position.y, hit.position.z, 1);
		vec4 c4(center.x, center.y, center.z, 0);
		hit.normal = normalize(gradf(cross - c4));
		hit.material = material;
		return hit;
	}

	Hit otherIntersect(const Ray& ray) {
		Hit hit;

		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 S(ray.start.x - center.x, ray.start.y - center.y, ray.start.z - center.z, 1);

		float a = f(D);
		float b = dot(S * Q, D) + dot(D * Q, S);
		float c = f(S);

		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;

		hit.t = t1;
		hit.position = ray.start + ray.dir * hit.t;
		vec4 cross(hit.position.x, hit.position.y, hit.position.z, 1);
		vec4 c4(center.x, center.y, center.z, 0);
		hit.normal = normalize(gradf(cross - c4));
		hit.material = material;
		return hit;
	}

};

/* A specialisabb kvadratikus objektumoknak specialis osztalyokat is keszitettem, pusztan az egyszerubb inicializalas
erdekeben, hogy ne kelljen mindig a teljes matrixot megadni. */

struct Ellipsoid : public Quadrics {

	Ellipsoid(const vec3& _center, const vec3& _abc, Material* _material) {
		center = _center;
		Q = mat4(	1.0 / _abc.x, 0.0, 0.0, 0.0,
					0.0, 1.0 / _abc.y, 0.0, 0.0,
					0.0, 0.0, 1.0 / _abc.z, 0.0,
					0.0, 0.0, 0.0, -1.0);
		material = _material;
	}

};

struct Hiperboloid : public Quadrics {

	Hiperboloid(const vec3& _center, const vec3& _abc, Material* _material) {
		center = _center;
		Q = mat4(	1.0 / _abc.x, 0.0, 0.0, 0.0,
					0.0, 1.0 / _abc.y, 0.0, 0.0,
					0.0, 0.0, -1.0 / _abc.z, 0.0,
					0.0, 0.0, 0.0, -1.0);
		material = _material;
	}

};

struct Paraboloid : public Quadrics {

	Paraboloid(const vec3& _center, const vec3& _abc, Material* _material) {
		center = _center;
		Q = mat4(	_abc.x, 0.0, 0.0, 0.0,
					0.0, _abc.y, 0.0, 0.0,
					0.0, 0.0, 0.0, -0.5*_abc.z,
					0.0, 0.0, -0.5* _abc.z, 0.0);
		material = _material;
	}

};

struct Cylinder : public Quadrics {

	Cylinder(const vec3& _center, const vec2& _ab, float _r, Material* _material) {
		center = _center;
		Q = mat4(	_ab.x, 0.0, 0.0, 0.0,
					0.0, _ab.y, 0.0, 0.0,
					0.0, 0.0, 0.0, 0.0,
					0.0, 0.0, 0.0, -1.0 * _r * _r);
		material = _material;
	}

};

struct Plane : public Intersectable {
	vec4 plane;

	Plane(const vec4& _plane, Material* _material) {
		plane = _plane;
		material = _material;
	}

	Hit intersect(const Ray& ray) {

		Hit hit;

		float dotProduct = dot(ray.dir, vec3(plane.x, plane.y, plane.z));
		if (fabs(dotProduct) < epsilon) {
			return hit;
		}

		float t = ((-1.0) * (dot(plane, vec4(ray.start.x, ray.start.y, ray.start.z, 1.0)))) / (dot(plane, vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0.0)));
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalize(vec3(plane.x, plane.y, plane.z));
		hit.material = material;
		return hit;
	}

};

bool lineCrossesPlane(const vec3& line_dir, const vec4& plane_normal) {
	vec3 normal(plane_normal.x, plane_normal.y, plane_normal.z);
	if (fabs(dot(normal, line_dir)) < epsilon) {
		return false;
	}
	return true;
}

float distanceFromPlane(const vec3& p, const vec4& plane_normal) {
	vec4 p1(p.x, p.y, p.z, 1.0);
	float dotProduct = dot(p1, plane_normal);
	float lengthOfNormal = length(vec3(plane_normal.x, plane_normal.y, plane_normal.z));
	return dotProduct / lengthOfNormal;
}


/* Sikokkal vagott konvex kvadratikus felulet, a sikok lehetnek lyukak is.
Igy a programommal barmilyen konvex poligont konnyen letre tudok hozni. */
struct Surface : public Intersectable {
	std::vector<Plane*> planes;
	std::vector<bool> isHole;
	Quadrics qd;

	Surface(Quadrics _qd) {
		qd = _qd;
	}

	void addPlane(const vec4& _plane, bool _isHole, Material* _material) {
		Plane* p = new Plane(_plane, _material);
		planes.push_back(p);
		isHole.push_back(_isHole);
	}

	Hit intersect(const Ray& ray) {
		Hit hit1, hit2, empty;

		hit1 = qd.intersect(ray);
		hit2 = qd.otherIntersect(ray);
		bool hit2hole = false;
		
		if (hit1.t < 0) { // a sugar metszi a testet?
			return empty; 
		}

		for (int i = 0; i < planes.size(); ++i) { // minden sikra megnezzuk
			if (!lineCrossesPlane(ray.dir, planes[i]->plane)) { // ha nem metszi a sikot
				if (distanceFromPlane(ray.start, planes[i]->plane) > 0) { // ha kulso pontok a sugar pontjai
					return empty; // akkor uresseg
				}
			}
			else { // ha metszi a sikot 
				if (distanceFromPlane(ray.start, planes[i]->plane) < 0) { // ha hatulrol erkezik
					if (planes[i]->intersect(ray).t < hit1.t && planes[i]->intersect(ray).t > 0) { // ha a test elott metszi
						return empty;
					}
					else if (planes[i]->intersect(ray).t >= hit1.t && planes[i]->intersect(ray).t <= hit2.t) { // ha a testben metszi
						hit2 = planes[i]->intersect(ray);
						if (isHole[i]) {
							hit2hole = true;
						}
					}
				}
				else { // ha szembol
					if (planes[i]->intersect(ray).t < hit1.t && planes[i]->intersect(ray).t > 0); // ha a test elott metszi
					else if (planes[i]->intersect(ray).t > hit1.t&& planes[i]->intersect(ray).t < hit2.t) { // ha a testben metszi
						hit1 = planes[i]->intersect(ray);
						if (isHole[i]) {
							hit1 = hit2;
							if (hit2hole) {
								hit1 = empty;
							}
						}
					}
					else { // ha a test utan metszi
						return empty;
					}
				}
			}
		}
		return hit1;
	}
};

/* raytrace.cpp - bol*/
class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(const vec3& _eye, const vec3& _lookat, const vec3& vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
};

/* raytrace.cpp - bol*/
struct Light {
	vec3 direction;
	vec3 Le;

	Light(const vec3& _direction, const vec3& _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};


class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
	vec3 La_room; // szobaban levo ambiens feny, hogy az arnyekolt reszek ne legyenek teljesen sotetek
	vec2 randomPoints[nv]; // a random pontok a fenycso also lemezen
public:
	void build() {

		for (int i = 0; i < nv; ++i) {
			float pr = rnd();
			float p_angle = 2 * M_PI * rnd();
			randomPoints[i] = vec2(pr * cosf(p_angle), pr * sinf(p_angle));
		}

		vec3 eye = vec3(8, 0, 1), vup = vec3(0, 0, 1), lookat = vec3(0, 0, 0.5);
		float fov = 95 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.53f, 0.81f, 0.92f); // ambiens feny, kek mint az eg
		La_room = vec3(0.2f, 0.2f, 0.2f);

		vec3 lightDirection(0, 15, 10), Le(1150, 1150, 1150);
		lights.push_back(new Light(lightDirection, Le));

		vec3 ks_room(0.05, 0.05, 0.05);
		vec3 ks_object(1, 1, 1);

		vec3 kd_room(0.50f, 0.61f, 0.87f);
		vec3 kd_left(0.7f, 0.1f, 0.5f);
		vec3 kd_right(0.1f, 0.8f, 0.3f);

		vec3 n_silver(0.14, 0.16, 0.13), kappa_silver(4.1, 2.3, 3.1);
		vec3 n_gold(0.17, 0.35, 1.5), kappa_gold(3.1, 2.7, 1.9);

		Material* silver = new ReflectiveMaterial(n_silver, kappa_silver);
		Material* gold = new ReflectiveMaterial(n_gold, kappa_gold);
		Material* mat_room = new RoughMaterial(kd_room, ks_room, 100);
		Material* mat_left = new RoughMaterial(kd_left, ks_object, 50);
		Material* mat_right = new RoughMaterial(kd_right, ks_object, 75);

		Ellipsoid room(vec3(0,0,0), vec3(200, 200, 50), mat_room);
		Surface surf_room(room);
		surf_room.addPlane(vec4(0, 0, 1, (-1.0) * (sqrtf(49.75))), true, mat_room); // felso lyuk
		objects.push_back(new Surface(surf_room));

		Hiperboloid velux(vec3(0, 0, 6.0 / sqrtf(5.0) + sqrtf(49.75) - 12.0 / sqrtf(5.0)), vec3(1.0 / 5.0, 1.0 / 5.0, 9.0 / 5.0), silver);
		Surface surf_velux(velux);
		surf_velux.addPlane(vec4(0, 0, 1, (-1.0) * (sqrtf(49.75) + (12.0 / (sqrtf(5.0))))), true, silver); // felso lyuk
		surf_velux.addPlane(vec4(0, 0, -1, sqrtf(49.75)), true, silver); // also lyuk
		objects.push_back(new Surface(surf_velux));

		Cylinder cyll(vec3(-1.0, -5.5, 0.0), vec2(1, 1), 2.0, mat_left);
		Surface surfcyll(cyll);
		surfcyll.addPlane(vec4(0, 0.13, 1, (-1.0) * -3.0), false, mat_left); // felso hatar
		surfcyll.addPlane(vec4(0, -0.205, -1, (1.0) * -7.605), false, mat_left); // also hatar
		objects.push_back(new Surface(surfcyll));

		Paraboloid par(vec3(-7, 0,4), vec3(0.85, 0.85,-1.0), gold);
		Surface surf_par(par);
		surf_par.addPlane(vec4(-1.82, 0, -6.29, -49.65), false, gold);
		objects.push_back(new Surface(surf_par));

		Ellipsoid ell(vec3(-4, 7.0, -2.0), vec3(5, 5, 16), mat_right);
		Surface surf_ell(ell);
		objects.push_back(new Surface(surf_ell));
	}

	/* raytrace.cpp - bol */
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
			if (dbrender)printf("\nY= %d", Y);
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	/* raytrace.cpp - bol */
	Hit firstIntersect(const Ray& ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1.0);
		return bestHit;
	}

	/* raytrace.cpp - bol*/
	bool shadowIntersect(const Ray& ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(const Ray& ray, int depth = 0) {
		if (depth > depthOfRecursion) return La;
		
		Hit hit = firstIntersect(ray); // mit metszunk el eloszor, erre beallitjuk a hit strukturat
		if (hit.t < 0) {
			return La + lights[0]->Le * pow(dot(ray.dir, lights[0]->direction), 10); // ha semmit, akkor az ambiens fennyel terunk vissza (uresseget latunk), most az ambiens feny a napiranytol valo elteres fuggvenyeben valtozik
		}
		vec3 outRadiance(0, 0, 0); // alapbol sotet van (amikor semmit nem talalunk el, azt az esetet az elobb mar lefedtuk)
		
		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La_room; // a szobaban levo ambiens feny

			vec3 lightFromTube(0, 0, 0);

			for (int i = 0; i < nv; ++i) {
				
				vec3 diskP(randomPoints[i].x, randomPoints[i].y, sqrtf(49.75));
				vec3 fromDiskToP(hit.position - diskP);
				vec3 dirFromDisk(normalize(fromDiskToP));
				float cosThetaIn = dot(hit.normal, (-1.0) * dirFromDisk);
				
				if (cosThetaIn > 0) {
					float cosTheta = (dot(dirFromDisk, normalOfDisk)); 
					float dOmega = (r_velux * r_velux * M_PI * cosTheta) / (nv * length(fromDiskToP) * length(fromDiskToP));
					vec3 Lin = trace(Ray(hit.position + hit.normal * epsilon, (-1.0) * (dirFromDisk)), depth + 1);
					vec3 brdf = hit.material->kd;
					lightFromTube = lightFromTube + Lin * cosThetaIn * brdf * dOmega;
					vec3 halfway = normalize(-ray.dir + (-1.0) * dirFromDisk);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) lightFromTube = lightFromTube + lights[0]->Le * hit.material->ks * powf(cosDelta, hit.material->shininess) * dOmega;
				}
			}

			outRadiance = outRadiance + lightFromTube;
		}

		/* Kiadott videobol */
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1.0 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}

		return outRadiance;
	}
};

/* Az innentoli reszek majdnem teljesen a raytrace.cpp - bol szarmaznak */
GPUProgram gpuProgram;
Scene scene;

class FullScreenTexturedQuad {
	unsigned int vao;
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		unsigned int vbo;
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY){}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}
