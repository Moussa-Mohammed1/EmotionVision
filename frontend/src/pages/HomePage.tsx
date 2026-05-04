import { useState } from "react";
import api from "../api/axios";


type FacePrediction = {
    bbox: [number, number, number, number];
    emotion: string;
    confidence: number;
};

function Home() {
    const [file, setFile] = useState<File | null>(null)
    const [message, setMessage] = useState<string>("");
    const [faces, setFaces] = useState<FacePrediction[]>([]);
    const [loading, setLoading] = useState<Boolean>(false);
    const uploadFile = async () => {
        if (!file) return;
        setLoading(true)
        const form = new FormData();
        form.append("file", file);
        try {
            const res = await api.post('/detect-face', form, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            console.log(res?.data);
            setMessage(res?.data.message);
            setFaces(res?.data.faces ?? []);
        } catch (error) {
            setMessage(`Upload failed: ${error}`);
            setFaces([]);
        }finally
        {
            setLoading(false)
        }
    }

    return (
        <div>
            <div>
                <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
                <button onClick={uploadFile} disabled={!file}>
                    {loading ? 'Sending...' : 'Send'}
                    
                </button>
            </div>
            <p className="pr">
                {message}
            </p>
            <div>
                {faces.map((face, index) => (
                    <p key={`${face.emotion}-${index}`}>
                        Face {index + 1}: {face.emotion} ({Math.round(face.confidence * 100)}%)
                    </p>
                ))}
            </div>

        </div>
    )
}
export default Home